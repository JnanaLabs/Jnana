"""
Generator
Takes retrieved chunks and a user query, builds a prompt, and calls the LLM.
Supports OpenAI GPT-4o-mini (default) and Google Gemini Flash.
"""

import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "gemini"

SYSTEM_PROMPT = """You are Jnana, an expert on Vedic scriptures including the Bhagavad-gita, 
Srimad-Bhagavatam, Caitanya-caritamrta, and other works by Srila Prabhupada.

Answer questions using ONLY the provided scripture passages. 
Always cite the specific verse reference (e.g. BG 2.47, SB 1.2.6).
If the answer is not in the provided passages, say so honestly.
Be respectful, clear, and spiritually illuminating in your responses."""


def format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a readable context block."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        ref = chunk.get("title") or f"{chunk.get('book_id')} {chunk.get('chapter')}.{chunk.get('verse')}"
        context_parts.append(f"[{i}] {ref}\n{chunk['text']}")
    return "\n\n---\n\n".join(context_parts)


def generate_openai(query: str, context: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Scripture passages:\n\n{context}\n\nQuestion: {query}"}
        ],
        temperature=0.3,
        max_tokens=1024
    )
    return response.choices[0].message.content


def generate_gemini(query: str, context: str) -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel(
        model_name=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        system_instruction=SYSTEM_PROMPT
    )
    prompt = f"Scripture passages:\n\n{context}\n\nQuestion: {query}"
    response = model.generate_content(prompt)
    return response.text


def generate(query: str, chunks: list[dict]) -> dict:
    """
    Generate an answer from retrieved chunks.

    Returns:
        dict with 'answer', 'sources' (list of references), and 'chunks_used'
    """
    if not chunks:
        return {
            "answer": "I could not find relevant scripture passages for your question. Please try rephrasing.",
            "sources": [],
            "chunks_used": 0
        }

    context = format_context(chunks)
    sources = [
        {"ref": c.get("title"), "url": c.get("url"), "translation": c.get("translation")}
        for c in chunks
    ]

    if LLM_PROVIDER == "gemini":
        answer = generate_gemini(query, context)
    else:
        answer = generate_openai(query, context)

    return {
        "answer": answer,
        "sources": sources,
        "chunks_used": len(chunks)
    }
