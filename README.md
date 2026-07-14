# 🕉️ Vedic RAG — Jnana Labs

A Retrieval-Augmented Generation (RAG) pipeline over Vedic scriptures from [Vedabase.io](https://vedabase.io), built by [Jnana Labs](https://github.com/JnanaLabs).

Ask questions about the **Bhagavad-gita, Srimad-Bhagavatam, Caitanya-caritamrta**, and more — get cited, scripture-grounded answers.

---

## Architecture

```
Vedabase.io → Scraper → Chunker → Embedder → Supabase pgvector
                                                    ↓
                                User Query → Retriever → Generator (LLM) → Answer + Citations
```

## Project Structure

```
vedic-rag/
├── scraper/
│   ├── vedabase_scraper.py    # Async Playwright scraper
│   └── books_index.json       # Scripture catalog
├── pipeline/
│   ├── chunker.py             # Verse-aware chunking
│   ├── embedder.py            # Embedding + Supabase upsert
│   └── ingest.py              # Full pipeline orchestrator
├── rag/
│   ├── retriever.py           # pgvector semantic search
│   └── generator.py           # LLM answer generation
├── api/
│   └── main.py                # FastAPI server
├── supabase/
│   └── schema.sql             # DB schema + RPC functions
├── data/                      # Created at runtime
│   ├── raw/                   # Scraped JSON per book
│   └── chunked/               # Chunked JSON per book
├── .env.example
└── requirements.txt
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 2. Configure environment

```bash
cp .env.example .env
# Fill in SUPABASE_URL, SUPABASE_KEY, OPENAI_API_KEY
```

### 3. Set up Supabase

Run the SQL in `supabase/schema.sql` in your Supabase SQL editor.

### 4. Run the ingestion pipeline

```bash
# Scrape, chunk, and embed a single book
python pipeline/ingest.py --book bg

# Or run everything
python pipeline/ingest.py --all

# Run steps individually
python scraper/vedabase_scraper.py --book bg
python pipeline/chunker.py --book bg
python pipeline/embedder.py --book bg
```

### 5. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`

### 6. Query the API

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the meaning of dharma in the Bhagavad-gita?"}'
```

## Embedding Models

| Model | Cost | Dimension | Quality |
|---|---|---|---|
| `text-embedding-3-small` (OpenAI) | ~$0.02/1M tokens | 1536 | ⭐⭐⭐⭐ |
| `BAAI/bge-small-en-v1.5` (local, free) | Free | 384 | ⭐⭐⭐ |

Set `EMBED_MODEL=local` in `.env` to use the free HuggingFace model (update vector dimension to 384 in schema.sql).

## Deployment

Deploy to **Railway** or **Render** (both free tier):
- Set environment variables in dashboard
- Start command: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`

## Scripture Coverage

| Book | ID |
|---|---|
| Bhagavad-gita As It Is | `bg` |
| Srimad-Bhagavatam | `sb` |
| Sri Caitanya-caritamrta | `cc` |
| Sri Isopanisad | `iso` |
| Nectar of Instruction | `noi` |
| Nectar of Devotion | `nod` |
| Brahma-samhita | `bns` |

---

Built with 🙏 by Jnana Labs
