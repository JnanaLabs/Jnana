"""
Generator
Takes retrieved chunks and a user query, builds a prompt, and generates an answer.

Provider priority (auto-detected from env):
  1. OpenAI  — if OPENAI_API_KEY is set
  2. Gemini  — fallback (free tier, generous limits)

Models configurable via .env:
  OPENAI_CHAT_MODEL   (default: gpt-4o-mini)
  GEMINI_CHAT_MODEL   (default: gemini-1.5-flash)
"""

import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

SYSTEM_PROMPT = """You are Jnana, an expert on Vedic scriptures including the Bhagavad-gita,
Srimad-Bhagavatam, Caitanya-caritamrta, and other works by Srila Prabhupada.

Answer questions using ONLY the provided scripture passages.
Always cite the specific verse reference (e.g. BG 2.47, SB 1.2.6).
If the answer is not in the provided passages, say so honestly.
Be respectful, clear, and spiritually illuminating in your responses."""


def _active_provider() -> str:
    """Auto-detect which LLM provider to use based on available API keys."""
    if os.getenv("OPENAI_API_KEY"):
        logger.debug("LLM provider: OpenAI")
        return "openai"
    if os.getenv("GEMINI_API_KEY"):
        logger.debug("LLM provider: Gemini (fallback)")
        return "gemini"
    raise EnvironmentError(
        "No LLM API key found. Set OPENAI_API_KEY or GEMINI_API_KEY in your .env file."
    )


def format_context(chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        ref = chunk.get("title") or f"{chunk.get('book_id')} {chunk.get('chapter')}.{chunk.get('verse')}"
        context_parts.append(f"[{i}] {ref}\n{chunk['text']}")
    return "\n\n---\n\n".join(context_parts)


def _generate_openai(query: str, context: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    logger.info(f"Generating with OpenAI model: {model}")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Scripture passages:\n\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.3,
        max_tokens=1024
    )
    return response.choices[0].message.content


def _generate_gemini(query: str, context: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model_name = os.getenv("GEMINI_CHAT_MODEL", "gemini-1.5-flash")
    logger.info(f"Generating with Gemini model: {model_name}")
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT
    )
    prompt = f"Scripture passages:\n\n{context}\n\nQuestion: {query}"
    response = model.generate_content(prompt)
    return response.text


def generate(query: str, chunks: list[dict]) -> dict:
    """
    Generate a grounded answer from retrieved scripture chunks.

    Returns:
        dict with keys: answer (str), sources (list), chunks_used (int), provider (str)
    """
    if not chunks:
        return {
            "answer": "I could not find relevant scripture passages for your question. Please try rephrasing.",
            "sources": [],
            "chunks_used": 0,
            "provider": "none"
        }

    context = format_context(chunks)
    sources = [
        {"ref": c.get("title"), "url": c.get("url"), "translation": c.get("translation")}
        for c in chunks
    ]

    provider = _active_provider()
    answer = _generate_openai(query, context) if provider == "openai" else _generate_gemini(query, context)

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks),
        "provider": provider
    }
