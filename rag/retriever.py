"""
Retriever
Performs semantic search over the vedic_chunks table in Supabase using pgvector.
Returns top-k chunks most relevant to the query.
"""

import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "openai")
DEFAULT_TOP_K = 6
DEFAULT_SIMILARITY_THRESHOLD = 0.75


def embed_query(query: str) -> list[float]:
    if EMBED_MODEL == "local":
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("BAAI/bge-small-en-v1.5")
        return model.encode([query], normalize_embeddings=True)[0].tolist()
    else:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.embeddings.create(model="text-embedding-3-small", input=[query])
        return response.data[0].embedding


def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    book_filter: str = None,
    similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD
) -> list[dict]:
    """
    Retrieve top-k relevant chunks for a query.

    Args:
        query: Natural language question
        top_k: Number of chunks to return
        book_filter: Optional book_id to restrict search (e.g. 'bg')
        similarity_threshold: Minimum cosine similarity score

    Returns:
        List of chunk dicts with text, metadata, and similarity score
    """
    from supabase import create_client
    supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

    query_embedding = embed_query(query)

    # Call Supabase RPC function for vector search
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
    logger.debug(f"Retrieved {len(chunks)} chunks for query: '{query[:60]}'")
    return chunks
