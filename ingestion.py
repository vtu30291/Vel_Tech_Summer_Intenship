"""
Document ingestion pipeline: PDF extraction, token-based chunking,
embedding generation, MySQL storage, and FAISS index insertion.
Runs in background threads to avoid blocking Flask request handlers.
"""

import os
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

    BUG FIX: Uses SELECT ... FOR UPDATE to serialize FAISS ID generation,
    preventing duplicate IDs when multiple documents are ingested concurrently.
    """
    logger.info(f"Starting ingestion process for material {material_id}")
    conn = None
    try:
        # Extract text
        text = extract_text_from_pdf(file_path)
        if not text.strip():
            raise ValueError("No text could be extracted from the PDF file.")

        # Chunk text
        chunks = semantic_chunking(text)
        if not chunks:
            raise ValueError("Document was empty or generated zero chunks.")

        logger.info(f"Generated {len(chunks)} chunks for material {material_id}")

        # Load embedding model
        model = get_embedding_model()

        # Generate embeddings
        embeddings = model.encode(chunks)
        embeddings = np.array(embeddings).astype("float32")

        # Connect to MySQL and perform transactions
        conn = get_db_connection()
        cursor = conn.cursor()

        # BUG FIX: Use SELECT ... FOR UPDATE to acquire a row-level lock,
        # preventing race conditions when two threads read MAX(id) simultaneously.
        # We lock the chunks table metadata row to serialize access.
        cursor.execute("SELECT COALESCE(MAX(id), 0) FROM chunks FOR UPDATE")
        start_id = cursor.fetchone()[0] + 1

        faiss_ids = []
        chunk_insert_data = []

        for idx, chunk_text in enumerate(chunks):
            faiss_id = start_id + idx
            faiss_ids.append(faiss_id)
            chunk_insert_data.append((material_id, idx, chunk_text, faiss_id))

        # Bulk insert chunks to MySQL
        insert_query = """
            INSERT INTO chunks (material_id, chunk_index, text_content, faiss_id)
            VALUES (%s, %s, %s, %s)
        """
        cursor.executemany(insert_query, chunk_insert_data)

        # Add vectors to FAISS index
        faiss_ids_arr = np.array(faiss_ids, dtype=np.int64)
        vector_store.add_vectors(embeddings, faiss_ids_arr)

        # Update material index_id, chunk_count, and status
        update_query = """
            UPDATE materials
            SET index_id = %s, chunk_count = %s, status = 'active'
            WHERE id = %s
        """
        cursor.execute(update_query, (str(start_id), len(chunks), material_id))

        conn.commit()
        cursor.close()
        logger.info(
            f"Successfully finalized ingestion and activated material {material_id}"
        )

    except Exception as e:
        logger.error(
            f"Failed to ingest document {material_id}: {e}", exc_info=True
        )
        # Set status to inactive on error
        try:
            if conn is None:
                conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE materials SET status = 'inactive' WHERE id = %s",
                (material_id,),
            )
            conn.commit()
            cursor.close()
        except Exception as rollback_err:
            logger.error(f"Failed to reset status on error: {rollback_err}")

    finally:
        if conn:
            try:
                conn.close()
            except Exception:
                pass


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
