"""
Document ingestion pipeline: PDF extraction, token-based chunking,
embedding generation, MySQL storage, and FAISS index insertion.
Runs in background threads to avoid blocking Flask request handlers.
"""

import os
import traceback
import numpy as np
import threading
import logging

import config
from db import get_db_connection
from vector_store import VectorStoreManager

logger = logging.getLogger(__name__)

_model = None
_model_lock = threading.Lock()


def get_embedding_model():
    """
    Lazy-load and cache the SentenceTransformer model (thread-safe singleton).
    BUG FIX: Removed redundant inner 'import os' that shadowed the module-level import.
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
    Extract text characters cleanly from private binary PDF files using PyMuPDF.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at {pdf_path}")

    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text("text")
    doc.close()
    return text


def semantic_chunking(
    text: str,
    chunk_size: int = config.CHUNK_SIZE,
    overlap: int = config.CHUNK_OVERLAP,
) -> list:
    """
    Semantic chunking logic that cuts raw text into overlapping spans
    of approximately 500 tokens (configurable via config.py).
    """
    model = get_embedding_model()
    tokenizer = model.tokenizer

    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return []

    chunks = []
    if len(tokens) <= chunk_size:
        # If smaller than chunk size, return single decoded text
        return [tokenizer.decode(tokens, skip_special_tokens=True)]

    start_idx = 0
    step = chunk_size - overlap

    while start_idx < len(tokens):
        end_idx = min(start_idx + chunk_size, len(tokens))
        chunk_tokens = tokens[start_idx:end_idx]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        # Only add non-empty chunks
        if chunk_text.strip():
            chunks.append(chunk_text)

        if end_idx == len(tokens):
            break
        start_idx += step

    return chunks


def process_document_ingestion(
    material_id: str, file_path: str, vector_store: VectorStoreManager
):
    """
    Ingests a PDF document: extracts text, chunks, vectorizes, saves to DB,
    saves to FAISS, and updates status to active.
    This runs in a background thread to prevent REST API blocks.

    Each major step is wrapped in its own try/except with full traceback logging
    so that any failure produces a complete error message in Render logs.
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
    # STEP 2: Semantic chunking
    # ------------------------------------------------------------------ #
    try:
        logger.info(f"[STEP 2] Running semantic chunking for material {material_id}")
        chunks = semantic_chunking(text)
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
    # STEP 3: Generate embeddings
    # ------------------------------------------------------------------ #
    try:
        logger.info(f"[STEP 3] Generating embeddings for material {material_id}")
        model = get_embedding_model()
        embeddings = model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")
        logger.info(
            f"[STEP 3 OK] Embeddings shape: {embeddings.shape} for material {material_id}"
        )
    except Exception:
        logger.exception(
            f"[STEP 3 FAILED] Embedding generation failed for material {material_id}.\n"
            + traceback.format_exc()
        )
        _mark_inactive(material_id)
        return

    # ------------------------------------------------------------------ #
    # STEP 4: Open DB connection with autocommit=False (explicit transaction)
    # ------------------------------------------------------------------ #
    try:
        logger.info(f"[STEP 4] Opening DB connection for material {material_id}")
        conn = get_db_connection()
        conn.autocommit = False  # Ensure explicit transaction control
        cursor = conn.cursor()
        logger.info(f"[STEP 4 OK] DB connection established for material {material_id}")
    except Exception:
        logger.exception(
            f"[STEP 4 FAILED] DB connection failed for material {material_id}.\n"
            + traceback.format_exc()
        )
        _mark_inactive(material_id)
        return

    # ------------------------------------------------------------------ #
    # STEP 5: Lock chunks table and determine FAISS IDs
    # ------------------------------------------------------------------ #
    try:
        logger.info(
            f"[STEP 5] Acquiring row-level lock and computing FAISS IDs for material {material_id}"
        )
        cursor.execute("SELECT COALESCE(MAX(id), 0) FROM chunks FOR UPDATE")
        row = cursor.fetchone()
        start_id = row[0] + 1
        logger.info(
            f"[STEP 5 OK] FAISS start_id={start_id} for material {material_id}"
        )
    except Exception:
        logger.exception(
            f"[STEP 5 FAILED] Failed to acquire lock / compute FAISS IDs for material {material_id}.\n"
            + traceback.format_exc()
        )
        try:
            conn.rollback()
        except Exception:
            pass
        _mark_inactive(material_id)
        return

    # ------------------------------------------------------------------ #
    # STEP 6: Bulk-insert chunks into MySQL
    # ------------------------------------------------------------------ #
    faiss_ids = []
    try:
        logger.info(
            f"[STEP 6] Bulk-inserting {len(chunks)} chunks into MySQL for material {material_id}"
        )
        chunk_insert_data = []
        for idx, chunk_text in enumerate(chunks):
            faiss_id = start_id + idx
            faiss_ids.append(faiss_id)
            chunk_insert_data.append((material_id, idx, chunk_text, faiss_id))

        insert_query = """
            INSERT INTO chunks (material_id, chunk_index, text_content, faiss_id)
            VALUES (%s, %s, %s, %s)
        """
        cursor.executemany(insert_query, chunk_insert_data)
        logger.info(
            f"[STEP 6 OK] Inserted {len(chunks)} chunk rows for material {material_id}"
        )
    except Exception:
        logger.exception(
            f"[STEP 6 FAILED] MySQL chunk insert failed for material {material_id}.\n"
            + traceback.format_exc()
        )
        try:
            conn.rollback()
        except Exception:
            pass
        _mark_inactive(material_id)
        return

    # ------------------------------------------------------------------ #
    # STEP 7: Add vectors to FAISS index
    # ------------------------------------------------------------------ #
    try:
        logger.info(
            f"[STEP 7] Adding {len(faiss_ids)} vectors to FAISS for material {material_id}"
        )
        faiss_ids_arr = np.array(faiss_ids, dtype=np.int64)
        vector_store.add_vectors(embeddings, faiss_ids_arr)
        logger.info(
            f"[STEP 7 OK] FAISS index updated. Total vectors in index: {vector_store.get_total_vectors()}"
        )
    except Exception:
        logger.exception(
            f"[STEP 7 FAILED] FAISS add_vectors failed for material {material_id}.\n"
            + traceback.format_exc()
        )
        try:
            conn.rollback()
        except Exception:
            pass
        _mark_inactive(material_id)
        return

    # ------------------------------------------------------------------ #
    # STEP 8: Update materials table — set index_id, chunk_count, status='active'
    # ------------------------------------------------------------------ #
    try:
        logger.info(
            f"[STEP 8] Updating materials table to 'active' for material {material_id} "
            f"(chunk_count={len(chunks)}, index_id={start_id})"
        )
        update_query = """
            UPDATE materials
            SET index_id = %s, chunk_count = %s, status = 'active'
            WHERE id = %s
        """
        cursor.execute(update_query, (str(start_id), len(chunks), material_id))
        rows_affected = cursor.rowcount
        logger.info(
            f"[STEP 8 OK] UPDATE materials affected {rows_affected} row(s) for material {material_id}"
        )
        if rows_affected == 0:
            logger.warning(
                f"[STEP 8 WARN] UPDATE matched 0 rows — material {material_id} may not exist in DB!"
            )
    except Exception:
        logger.exception(
            f"[STEP 8 FAILED] UPDATE materials failed for material {material_id}.\n"
            + traceback.format_exc()
        )
        try:
            conn.rollback()
        except Exception:
            pass
        _mark_inactive(material_id)
        return

    # ------------------------------------------------------------------ #
    # STEP 9: Commit the transaction
    # ------------------------------------------------------------------ #
    try:
        logger.info(f"[STEP 9] Committing transaction for material {material_id}")
        conn.commit()
        logger.info(
            f"[STEP 9 OK] Transaction committed successfully for material {material_id}"
        )
    except Exception:
        logger.exception(
            f"[STEP 9 FAILED] conn.commit() failed for material {material_id}.\n"
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

    # ------------------------------------------------------------------ #
    # STEP 10: Verify the DB record was persisted correctly
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
                f"[STEP 10 VERIFY] material_id={material_id} -> "
                f"status={record['status']}, chunk_count={record['chunk_count']}, "
                f"index_id={record['index_id']}"
            )
        else:
            logger.warning(
                f"[STEP 10 WARN] Could not find material {material_id} in DB after commit!"
            )
    except Exception:
        logger.exception(
            f"[STEP 10 WARN] Post-commit verification query failed for material {material_id}.\n"
            + traceback.format_exc()
        )

    logger.info(
        f"[INGESTION COMPLETE] material_id={material_id} successfully activated."
    )

    # Close the main connection
    try:
        conn.close()
    except Exception:
        pass


def _mark_inactive(material_id: str):
    """
    Attempts to set the material status back to 'inactive' after a pipeline failure.
    Uses a fresh connection to avoid reusing a broken transaction.
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
