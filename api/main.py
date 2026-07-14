"""
Jnana — Vedic RAG API
FastAPI backend exposing the RAG pipeline.

Endpoints:
    POST /query    — Ask a question about Vedic scriptures
    GET  /health   — Health check
    GET  /books    — List available scripture books

Run locally:
    uvicorn api.main:app --reload --port 8000

API docs:
    http://localhost:8000/docs
"""

import os
import json
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

from rag.retriever import retrieve
from rag.generator import generate


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm up Supabase read client on startup."""
    from core.supabase_client import get_read_client
    try:
        get_read_client()
        logger.info("Supabase read client initialised.")
    except Exception as e:
        logger.error(f"Failed to initialise Supabase client: {e}")
    yield


app = FastAPI(
    title="Jnana — Vedic Scripture RAG",
    description=(
        "Ask questions about Vedic scriptures — Bhagavad-gita, Srimad-Bhagavatam, "
        "Caitanya-caritamrta, and more. Answers are grounded in actual verse passages."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS
allowed_origins = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ────────────────────── Schemas ──────────────────────

class QueryRequest(BaseModel):
    question:    str  = Field(..., min_length=3, max_length=1000,
                               description="Your question about Vedic scriptures")
    book_filter: str  = Field(None,
                               description="Restrict search to one book (e.g. 'bg', 'sb', 'cc')")
    top_k:       int  = Field(6, ge=1, le=20,
                               description="Number of passages to retrieve (1–20)")


class SourceItem(BaseModel):
    ref:         str
    url:         str
    translation: str


class QueryResponse(BaseModel):
    question:    str
    answer:      str
    sources:     list[SourceItem]
    chunks_used: int
    provider:    str  # "openai" | "gemini" | "none"


# ────────────────────── Routes ───────────────────────

@app.get("/health", tags=["Meta"])
async def health():
    """Returns service status. Use this to verify the API is running."""
    return {"status": "ok", "service": "jnana-vedic-rag", "version": "0.1.0"}


@app.get("/books", tags=["Meta"])
async def list_books():
    """Returns the list of available scripture books."""
    try:
        with open("scraper/books_index.json", encoding="utf-8") as f:
            index = json.load(f)
        return {"books": index["books"]}
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="books_index.json not found")


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query(request: QueryRequest):
    """
    Ask a question about Vedic scriptures.
    Returns a cited answer grounded in actual scripture passages.
    """
    logger.info(f"Query: '{request.question[:80]}' | book_filter={request.book_filter} | top_k={request.top_k}")
    try:
        chunks = retrieve(
            query=request.question,
            top_k=request.top_k,
            book_filter=request.book_filter,
        )
        result = generate(query=request.question, chunks=chunks)
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=[SourceItem(**s) for s in result["sources"]],
            chunks_used=result["chunks_used"],
            provider=result["provider"],
        )
    except EnvironmentError as e:
        logger.error(f"Configuration error: {e}")
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.exception(f"Unexpected error processing query: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
