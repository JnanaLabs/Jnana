"""
Retriever
Performs semantic search over vedic_chunks in Supabase pgvector.

Uses the same embedding provider that was used during ingestion.
Make sure your EMBED_MODEL and API keys match what you used in embedder.py.
"""

import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

DEFAULT_TOP_K = 6
DEFAULT_SIMILARITY_THRESHOLD = 0.75


def _embed_query(query: str) -> list[float]:
    """Embed query using the same provider used during ingestion."""
    if os.getenv("EMBED_MODEL") == "local":
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return m.encode([query], normalize_embeddings=True)[0].tolist()

    if os.getenv("OPENAI_API_KEY"):
        from openai import OpenAI
        model = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
        logger.debug(f"Query embedding via OpenAI: {model}")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(model=model, input=[query])
        return response.data[0].embedding

    if os.getenv("GEMINI_API_KEY"):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = os.getenv("GEMINI_EMBED_MODEL", "models/text-embedding-004")
        logger.debug(f"Query embedding via Gemini: {model}")
        res = genai.embed_content(model=model, content=query, task_type="retrieval_query")
        return res["embedding"]

    # Last resort: local
    from sentence_transformers import SentenceTransformer
    logger.warning("No API keys — using local embedding model.")
    m = SentenceTransformer("BAAI/bge-small-en-v1.5")
    return m.encode([query], normalize_embeddings=True)[0].tolist()


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    book_filter: str = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> list[dict]:
    from supabase import create_client
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    query_embedding = _embed_query(query)

    params = {
        "query_embedding": query_embedding,
        "match_count": top_k,
        "similarity_threshold": similarity_threshold
    }
    if book_filter:
        params["filter_book_id"] = book_filter

    rpc_name = "match_vedic_chunks_filtered" if book_filter else "match_vedic_chunks"
    result = supabase.rpc(rpc_name, params).execute()

    chunks = result.data or []
    logger.debug(f"Retrieved {len(chunks)} chunks for: '{query[:60]}'")
    return chunks
