"""
Document ingestion pipeline: PDF extraction, token-based chunking,
embedding generation, MySQL storage, and FAISS index insertion.
Runs in background threads to avoid blocking Flask request handlers.

Memory-optimised for Render free-tier (512 MB):
  - Embeddings are generated in small batches (EMBED_BATCH_SIZE chunks at a time).
  - Each batch is written to FAISS immediately, then freed with del + gc.collect().
  - The large raw text string is released after chunking.
  - torch.no_grad() prevents PyTorch from allocating gradient buffers.
  - Only one SentenceTransformer singleton is kept in memory at any time.
"""

import gc
import os
import traceback
import numpy as np
import threading
import logging

import config
from db import get_db_connection
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

# Batch size for embedding generation.
# 2 chunks at a time keeps peak RAM well under 512 MB on Render free-tier.
# Increase to 4 only if you are on a paid instance with more headroom.
EMBED_BATCH_SIZE: int = 2

_model = None
_model_lock = threading.Lock()


def get_embedding_model():
    """
    Lazy-load and cache the SentenceTransformer model (thread-safe singleton).
    Only one model instance ever lives in memory.
    """
    global _model
    with _model_lock:
        if _model is None:
            logger.info(
                f"Loading SentenceTransformer model '{config.EMBEDDING_MODEL_NAME}'..."
            )
            os.environ["HF_HUB_OFFLINE"] = "1"
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
            logger.info("SentenceTransformer model loaded successfully.")
        return _model


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text characters cleanly from a PDF using PyMuPDF.
    The fitz.Document is explicitly closed to free native memory immediately.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    doc.close()
    del doc
    return text


def semantic_chunking(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list:
    """
    Cuts raw text into overlapping token-spans (~500 tokens each).
    Returns a plain Python list of strings; the encoded token list is freed
    before returning to avoid keeping two copies of the document in memory.
    """
    model = get_embedding_model()
    tokenizer = model.tokenizer

    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return []

    chunks = []
    if len(tokens) <= chunk_size:
        single = tokenizer.decode(tokens, skip_special_tokens=True)
        del tokens
        return [single]

    start_idx = 0
    step = chunk_size - overlap

    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        del chunk_tokens  # free slice immediately
        if chunk_text.strip():
            chunks.append(chunk_text)
        if end_idx == len(tokens):
            break
        start_idx += step

    del tokens  # free the full token list now that chunking is done
    return chunks


def _encode_batch_no_grad(model, batch: list) -> np.ndarray:
    """
    Encode a small list of strings under torch.no_grad() to prevent PyTorch
    from allocating gradient buffers. Returns a float32 NumPy array.
    """
    try:
        import torch
        with torch.no_grad():
            vecs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    except ImportError:
        # torch not directly importable in some environments; fall back gracefully
        vecs = model.encode(batch, convert_to_numpy=True, show_progress_bar=False)
    return np.array(vecs, dtype="float32")


def process_document_ingestion(
    material_id: str, file_path: str, vector_store: VectorStoreManager
):
    """
    Ingests a PDF document in a memory-efficient streaming fashion:
      1. Extract text → release PDF
      2. Chunk text → release raw text string
      3. Open DB transaction + acquire FAISS ID range
      4. For each small batch of chunks:
           a. Encode with torch.no_grad()
           b. Insert chunk rows into MySQL
           c. Write vectors to FAISS
           d. Free the batch tensors (del + gc.collect())
      5. UPDATE materials → commit transaction
      6. Verify DB record

    Each major step is wrapped in its own try/except with full traceback
    so that any failure is fully visible in Render logs.
    """
    logger.info(
        f"[INGESTION START] material_id={material_id} | file_path={file_path}"
    )
    conn = None

    # ------------------------------------------------------------------ #
    # STEP 1: Extract text from PDF
    # ------------------------------------------------------------------ #
    try:
        logger.info(f"[STEP 1] Extracting text from PDF for material {material_id}")
        text = extract_text_from_pdf(file_path)
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF file.")
        logger.info(
            f"[STEP 1 OK] Extracted {len(text)} characters from {file_path}"
        )
    except Exception:
        logger.exception(
            f"[STEP 1 FAILED] PDF text extraction failed for material {material_id}.\n"
            + traceback.format_exc()
        )
        _mark_inactive(material_id)
        return

    # ------------------------------------------------------------------ #
    # STEP 2: Semantic chunking (release raw text immediately after)
    # ------------------------------------------------------------------ #
    try:
        logger.info(f"[STEP 2] Running semantic chunking for material {material_id}")
        chunks = semantic_chunking(text)
        # Release the large raw text string — chunks hold all we need now.
        del text
        gc.collect()
        if not chunks:
            raise ValueError("Document was empty or generated zero chunks.")
        logger.info(
            f"[STEP 2 OK] Generated {len(chunks)} chunks for material {material_id}"
        )
    except Exception:
        logger.exception(
            f"[STEP 2 FAILED] Chunking failed for material {material_id}.\n"
            + traceback.format_exc()
        )
        _mark_inactive(material_id)
        return

    # ------------------------------------------------------------------ #
    # STEP 3: Open DB connection and begin explicit transaction
    # ------------------------------------------------------------------ #
    try:
        logger.info(f"[STEP 3] Opening DB connection for material {material_id}")
        conn = get_db_connection()
        conn.autocommit = False  # explicit transaction control
        cursor = conn.cursor()
        logger.info(f"[STEP 3 OK] DB connection established for material {material_id}")
    except Exception:
        logger.exception(
            f"[STEP 3 FAILED] DB connection failed for material {material_id}.\n"
            + traceback.format_exc()
        )
        _mark_inactive(material_id)
        return

    # ------------------------------------------------------------------ #
    # STEP 4: Lock chunks table and determine starting FAISS ID
    # ------------------------------------------------------------------ #
    try:
        logger.info(
            f"[STEP 4] Acquiring row-level lock and computing FAISS start_id "
            f"for material {material_id}"
        )
        cursor.execute("SELECT COALESCE(MAX(id), 0) FROM chunks FOR UPDATE")
        row = cursor.fetchone()
        start_id = row[0] + 1
        logger.info(
            f"[STEP 4 OK] FAISS start_id={start_id} for material {material_id}"
        )
    except Exception:
        logger.exception(
            f"[STEP 4 FAILED] Failed to acquire lock / compute FAISS IDs "
            f"for material {material_id}.\n" + traceback.format_exc()
        )
        try:
            conn.rollback()
        except Exception:
            pass
        _mark_inactive(material_id)
        return

    # ------------------------------------------------------------------ #
    # STEP 5: Stream through chunks in small batches — encode → DB → FAISS → free
    # ------------------------------------------------------------------ #
    total_chunks = len(chunks)
    model = get_embedding_model()
    insert_query = """
        INSERT INTO chunks (material_id, chunk_index, text_content, faiss_id)
        VALUES (%s, %s, %s, %s)
    """

    try:
        logger.info(
            f"[STEP 5] Processing {total_chunks} chunks in batches of "
            f"{EMBED_BATCH_SIZE} for material {material_id}"
        )

        for batch_start in range(0, total_chunks, EMBED_BATCH_SIZE):
            batch_end = min(batch_start + EMBED_BATCH_SIZE, total_chunks)
            batch_texts = chunks[batch_start:batch_end]
            batch_indices = list(range(batch_start, batch_end))
            batch_faiss_ids = [start_id + i for i in batch_indices]

            logger.info(
                f"[STEP 5] Encoding batch chunks[{batch_start}:{batch_end}] "
                f"(faiss_ids {batch_faiss_ids[0]}..{batch_faiss_ids[-1]}) "
                f"for material {material_id}"
            )

            # 5a: Encode under no_grad — minimal memory footprint
            try:
                batch_embeddings = _encode_batch_no_grad(model, batch_texts)
            except Exception:
                logger.exception(
                    f"[STEP 5 FAILED] Embedding failed for batch "
                    f"chunks[{batch_start}:{batch_end}] material {material_id}.\n"
                    + traceback.format_exc()
                )
                try:
                    conn.rollback()
                except Exception:
                    pass
                _mark_inactive(material_id)
                return

            # 5b: Insert chunk rows into MySQL
            try:
                chunk_rows = [
                    (material_id, batch_indices[i], batch_texts[i], batch_faiss_ids[i])
                    for i in range(len(batch_texts))
                ]
                cursor.executemany(insert_query, chunk_rows)
            except Exception:
                logger.exception(
                    f"[STEP 5 FAILED] MySQL insert failed for batch "
                    f"chunks[{batch_start}:{batch_end}] material {material_id}.\n"
                    + traceback.format_exc()
                )
                try:
                    conn.rollback()
                except Exception:
                    pass
                _mark_inactive(material_id)
                return

            # 5c: Write this batch of vectors to FAISS immediately
            try:
                faiss_ids_arr = np.array(batch_faiss_ids, dtype=np.int64)
                vector_store.add_vectors(batch_embeddings, faiss_ids_arr)
                logger.info(
                    f"[STEP 5] Batch chunks[{batch_start}:{batch_end}] written to FAISS. "
                    f"Index total: {vector_store.get_total_vectors()}"
                )
            except Exception:
                logger.exception(
                    f"[STEP 5 FAILED] FAISS write failed for batch "
                    f"chunks[{batch_start}:{batch_end}] material {material_id}.\n"
                    + traceback.format_exc()
                )
                try:
                    conn.rollback()
                except Exception:
                    pass
                _mark_inactive(material_id)
                return

            # 5d: Explicitly free this batch's memory before next iteration
            del batch_embeddings, batch_texts, faiss_ids_arr, chunk_rows
            gc.collect()

        logger.info(
            f"[STEP 5 OK] All {total_chunks} chunks processed for material {material_id}"
        )

    finally:
        # Release the full chunks list now that all batches are done (or failed)
        del chunks
        gc.collect()

    # ------------------------------------------------------------------ #
    # STEP 6: UPDATE materials — set index_id, chunk_count, status='active'
    # ------------------------------------------------------------------ #
    try:
        logger.info(
            f"[STEP 6] Updating materials table to 'active' for material {material_id} "
            f"(chunk_count={total_chunks}, index_id={start_id})"
        )
        update_query = """
            UPDATE materials
            SET index_id = %s, chunk_count = %s, status = 'active'
            WHERE id = %s
        """
        cursor.execute(update_query, (str(start_id), total_chunks, material_id))
        rows_affected = cursor.rowcount
        logger.info(
            f"[STEP 6 OK] UPDATE materials affected {rows_affected} row(s) "
            f"for material {material_id}"
        )
        if rows_affected == 0:
            logger.warning(
                f"[STEP 6 WARN] UPDATE matched 0 rows — "
                f"material {material_id} may not exist in DB!"
            )
    except Exception:
        logger.exception(
            f"[STEP 6 FAILED] UPDATE materials failed for material {material_id}.\n"
            + traceback.format_exc()
        )
        try:
            conn.rollback()
        except Exception:
            pass
        _mark_inactive(material_id)
        return

    # ------------------------------------------------------------------ #
    # STEP 7: Commit the transaction
    # ------------------------------------------------------------------ #
    try:
        logger.info(f"[STEP 7] Committing transaction for material {material_id}")
        conn.commit()
        logger.info(
            f"[STEP 7 OK] Transaction committed successfully for material {material_id}"
        )
    except Exception:
        logger.exception(
            f"[STEP 7 FAILED] conn.commit() failed for material {material_id}.\n"
            + traceback.format_exc()
        )
        try:
            conn.rollback()
        except Exception:
            pass
        _mark_inactive(material_id)
        return
    finally:
        try:
            cursor.close()
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # STEP 8: Post-commit verification SELECT
    # ------------------------------------------------------------------ #
    try:
        verify_conn = get_db_connection()
        verify_cursor = verify_conn.cursor(dictionary=True)
        verify_cursor.execute(
            "SELECT id, status, chunk_count, index_id FROM materials WHERE id = %s",
            (material_id,),
        )
        record = verify_cursor.fetchone()
        verify_cursor.close()
        verify_conn.close()
        if record:
            logger.info(
                f"[STEP 8 VERIFY] material_id={material_id} -> "
                f"status={record['status']}, chunk_count={record['chunk_count']}, "
                f"index_id={record['index_id']}"
            )
        else:
            logger.warning(
                f"[STEP 8 WARN] Could not find material {material_id} in DB after commit!"
            )
    except Exception:
        logger.exception(
            f"[STEP 8 WARN] Post-commit verification failed for material {material_id}.\n"
            + traceback.format_exc()
        )

    logger.info(
        f"[INGESTION COMPLETE] material_id={material_id} successfully activated."
    )


def _mark_inactive(material_id: str):
    """
    Marks the material as 'inactive' after a pipeline failure.
    Always uses a fresh DB connection to avoid inheriting a broken transaction.
    """
    logger.info(
        f"[ROLLBACK] Attempting to mark material {material_id} as inactive..."
    )
    try:
        rollback_conn = get_db_connection()
        rollback_conn.autocommit = False
        rollback_cursor = rollback_conn.cursor()
        rollback_cursor.execute(
            "UPDATE materials SET status = 'inactive' WHERE id = %s",
            (material_id,),
        )
        rollback_conn.commit()
        rollback_cursor.close()
        rollback_conn.close()
        logger.info(
            f"[ROLLBACK OK] material {material_id} marked inactive after failure."
        )
    except Exception:
        logger.exception(
            f"[ROLLBACK FAILED] Could not mark material {material_id} as inactive.\n"
            + traceback.format_exc()
        )


def start_async_ingestion(
    material_id: str, file_path: str, vector_store: VectorStoreManager
):
    """
    Launches document ingestion in a background thread.
    """
    thread = threading.Thread(
        target=process_document_ingestion,
        args=(material_id, file_path, vector_store),
        daemon=True,
    )
    thread.start()
    return thread
