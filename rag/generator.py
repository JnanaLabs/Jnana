"""
Generator
Builds a grounded answer from retrieved scripture chunks using an LLM.

Provider priority (auto-detected from env keys):
  1. OpenAI  — if OPENAI_API_KEY is set  (model: OPENAI_CHAT_MODEL, default gpt-4o-mini)
  2. Gemini  — fallback                  (model: GEMINI_CHAT_MODEL, default gemini-1.5-flash)

Raises EnvironmentError if neither key is present.
"""

import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

SYSTEM_PROMPT = """You are Jnana, a knowledgeable and respectful guide to Vedic scriptures.
You have deep knowledge of the Bhagavad-gita, Srimad-Bhagavatam, Caitanya-caritamrta,
and other works by Srila Prabhupada.

Rules:
- Answer using ONLY the provided scripture passages. Do not add outside knowledge.
- Always cite the specific verse reference (e.g. BG 2.47, SB 1.2.6, CC Adi 1.1).
- If the provided passages do not contain a clear answer, say so honestly.
- Be respectful, clear, and spiritually illuminating.
- Keep answers concise but complete."""


def _active_provider() -> str:
    if os.getenv("OPENAI_API_KEY"):
        return "openai"
    if os.getenv("GEMINI_API_KEY"):
        return "gemini"
    raise EnvironmentError(
        "No LLM API key found. Set OPENAI_API_KEY or GEMINI_API_KEY in your .env file."
    )


def _format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        ref = chunk.get("title") or f"{chunk.get('book_id', '').upper()} {chunk.get('chapter')}.{chunk.get('verse')}"
        parts.append(f"[{i}] {ref}\n{chunk['text']}")
    return "\n\n---\n\n".join(parts)


def _generate_openai(query: str, context: str) -> str:
    from openai import OpenAI
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    logger.info(f"Generating with OpenAI: {model}")
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": f"Scripture passages:\n\n{context}\n\nQuestion: {query}"},
        ],
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


def _generate_gemini(query: str, context: str) -> str:
    import google.generativeai as genai
    model_name = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
    logger.info(f"Generating with Gemini: {model_name}")
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT,
    )
    prompt = f"Scripture passages:\n\n{context}\n\nQuestion: {query}"
    response = model.generate_content(prompt)
    return response.text


def generate(query: str, chunks: list[dict]) -> dict:
    """
    Generate a cited, grounded answer from retrieved scripture chunks.

    Returns:
        {
            answer:       str   — the LLM response
            sources:      list  — [{ref, url, translation}, ...]
            chunks_used:  int   — number of chunks passed to LLM
            provider:     str   — "openai" | "gemini" | "none"
        }
    """
    if not chunks:
        return {
            "answer": "I could not find relevant scripture passages for your question. Please try rephrasing.",
            "sources": [],
            "chunks_used": 0,
            "provider": "none",
        }

    context = _format_context(chunks)
    sources = [
        {
            "ref":         c.get("title") or f"{c.get('book_id','').upper()} {c.get('chapter')}.{c.get('verse')}",
            "url":         c.get("url") or "",
            "translation": c.get("translation") or "",
        }
        for c in chunks
    ]

    provider = _active_provider()
    answer = _generate_openai(query, context) if provider == "openai" else _generate_gemini(query, context)

    return {
        "answer":      answer,
        "sources":     sources,
        "chunks_used": len(chunks),
        "provider":    provider,
    }
