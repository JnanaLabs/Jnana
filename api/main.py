"""
Jnana Labs — Vedic RAG API
FastAPI backend exposing the RAG pipeline.

Endpoints:
    POST /query          — Ask a question, get an answer with citations
    GET  /health         — Health check
    GET  /books          — List available books
"""

import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Import RAG modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rag.retriever import retrieve
from rag.generator import generate

app = FastAPI(
    title="Jnana Labs — Vedic RAG API",
    description="Ask questions about Vedic scriptures (Bhagavad-gita, Srimad-Bhagavatam, etc.) powered by RAG.",
    version="0.1.0"
)

# CORS — allow all origins for now (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000, description="Your question about Vedic scriptures")
    book_filter: str = Field(None, description="Optional: limit search to a specific book ID (e.g. 'bg', 'sb')")
    top_k: int = Field(6, ge=1, le=20, description="Number of passages to retrieve")


class Source(BaseModel):
    ref: str
    url: str
    translation: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    chunks_used: int
    question: str


@app.get("/health")
async def health():
    return {"status": "ok", "service": "jnana-labs-vedic-rag"}


@app.get("/books")
async def list_books():
    with open("scraper/books_index.json") as f:
        index = json.load(f)
    return {"books": index["books"]}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    logger.info(f"Query: '{request.question}' | book_filter={request.book_filter}")
    try:
        chunks = retrieve(
            query=request.question,
            top_k=request.top_k,
            book_filter=request.book_filter
        )
        result = generate(query=request.question, chunks=chunks)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"],
            chunks_used=result["chunks_used"],
            question=request.question
        )
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))
