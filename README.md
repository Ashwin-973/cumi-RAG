# CUMI Abrasives AI Agent

A full-stack RAG (Retrieval-Augmented Generation) application that lets users ask natural-language questions about [CUMI Abrasives](https://cumiabrasive.in) products and get grounded, conversational answers backed by the full product catalogue — no hallucinations, no guessing.

---

## What it does

- Scrapes the entire CUMI Abrasives product catalogue using Playwright
- Embeds and indexes every product into Pinecone using `llama-text-embed-v2`
- On every query: performs dense vector search → cross-encoder reranking → LLM answer generation
- Returns a structured response: conversational answer + ranked product cards + follow-up suggestions
- Displays everything in a chat-style React UI with a scrollable product carousel

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE  (one-time)                    │
│                                                                     │
│   cumiabrasive.in  ──►  scrape_site.py  ──►  products.xlsx         │
│                         (Playwright)                                │
│                                                                     │
│   products.xlsx  ──►  create_index.py  ──►  Pinecone Index         │
│                        (embed + upsert)     cumi-abrasives-index    │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        BACKEND  (FastAPI)                           │
│                                                                     │
│  POST /api/ask                                                      │
│       │                                                             │
│       ▼                                                             │
│  [ retriever.py ]  ── Stage 1: embed query  ──►  Pinecone          │
│                        llama-text-embed-v2       top-30 candidates  │
│                    ── Stage 2: rerank        ──►  bge-reranker-v2   │
│                        cross-encoder              top-k results     │
│       │                                                             │
│       ▼                                                             │
│  [ prompt.py ]  ── inject product context into grounded prompt     │
│       │                                                             │
│       ▼                                                             │
│  [ llm.py ]  ── Ollama (llama3.2)  ──►  JSON output:              │
│                  local inference        { conversational_response,  │
│                                           suggested_follow_ups }    │
│       │                                                             │
│       ▼                                                             │
│  [ chain.py ]  ── merge LLM text + retriever product data          │
│       │            (LLM never touches product fields)              │
│       ▼                                                             │
│  QueryResponse  { conversational_response,                         │
│                   recommended_products[],                           │
│                   suggested_follow_ups[] }                          │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                        FRONTEND  (React + Vite)                     │
│                                                                     │
│  useChat hook  ──►  POST /api/ask  ──►  ECommerceRagResponse       │
│                                          ├─ conversational bubble   │
│                                          ├─ product card carousel   │
│                                          └─ follow-up pills         │
└─────────────────────────────────────────────────────────────────────┘
```

### Key architectural decisions

| Decision | Reason |
|---|---|
| **LLM generates text only** | Product data (specs, URLs, brand) comes directly from Pinecone metadata — the LLM never touches it, eliminating hallucination on facts |
| **Two-stage retrieval** | Wide dense search (top-30) followed by BGE cross-encoder reranking (top-5) gives high recall + high precision |
| **Pinecone Inference API** | Embedding and reranking are both handled server-side by Pinecone — no local GPU needed for those steps |
| **Ollama for generation** | Fully local LLM inference — no OpenAI API costs, no data leaves the machine |
| **Pydantic v2 schemas** | Request/response validation is enforced at the API boundary; the frontend can rely on a stable shape |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Scraping | Python, Playwright (async) |
| Data | Pandas, OpenPyXL |
| Vector DB | Pinecone Serverless (`us-east-1`) |
| Embedding | `llama-text-embed-v2` (1024-dim) via Pinecone Inference |
| Reranking | `bge-reranker-v2-m3` via Pinecone Inference |
| LLM | Ollama — `llama3.2` (local) |
| Backend | Python, FastAPI, Uvicorn, Pydantic v2 |
| Frontend | React 19, Vite 7, Tailwind CSS v4, Lucide icons |

---

## Project Structure

```
cumi-RAG/
├── scripts/
│   ├── scrape_site.py        # Playwright scraper → products.xlsx
│   └── create_index.py       # Embeds + upserts products into Pinecone
│
├── data/
│   ├── raw/                  # Raw scrape output
│   └── processed/
│       └── products.xlsx     # Cleaned product data (input to indexer)
│
├── backend/
│   └── app/
│       ├── main.py           # FastAPI app entry point
│       ├── api/routes/
│       │   └── query.py      # POST /api/ask
│       ├── core/
│       │   ├── config.py     # Pydantic settings (env vars)
│       │   ├── llm.py        # Ollama HTTP client
│       │   └── vectorstore.py# Cached Pinecone client + index handle
│       ├── models/
│       │   └── schema.py     # QueryRequest / QueryResponse schemas
│       ├── rag/
│       │   ├── chain.py      # Orchestrates retrieve → prompt → generate
│       │   ├── prompt.py     # Builds grounded prompt for the LLM
│       │   └── retriever.py  # Dense search + BGE reranking
│       └── utils/
│           └── metadata_filters.py  # Pinecone metadata filter builders
│
└── frontend/
    └── src/
        ├── App.jsx
        ├── lib/api.js              # fetch wrapper for /api/ask
        ├── hooks/useChat.jsx       # chat state + sendQuery logic
        └── components/
            ├── WelcomeScreen.jsx        # Landing screen with suggestion chips
            ├── ChatPage.jsx             # Main chat layout
            ├── ChatInput.jsx            # Query input bar
            ├── MessageThread.jsx        # Renders message history
            ├── ECommerceRagResponse.jsx # 3-layer response: bubble + carousel + pills
            └── ProductCard.jsx          # Individual product card
```

---

## Setup

### Prerequisites

- Python 3.11+
- Node.js 18+
- [Ollama](https://ollama.com) running locally with `llama3.2` pulled
- A [Pinecone](https://app.pinecone.io) account with an API key

### 1. Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r ../requirements.txt
```

Create a `.env` file in `backend/`:

```env
PINECODE_DEFAULT_API_KEY=your_pinecone_api_key
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.2
```

Start the server:

```bash
uvicorn app.main:app --reload
```

### 2. Index (run once)

```bash
cd scripts
python create_index.py
```

This embeds all products from `data/processed/products.xlsx` and upserts them into Pinecone. Comment out `run_ingestion()` after the first run to avoid duplicate vectors.

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

The app will be available at `http://localhost:5173`. Set `VITE_API_URL` in a `.env` file if your backend runs on a different port.

---

## API

### `POST /api/ask`

**Request**
```json
{
  "query": "cutting blade for tiles",
  "top_k": 5,
  "brand_filter": "CUMI",
  "category_filter": "Cutting Blades"
}
```

**Response**
```json
{
  "conversational_response": "For tile cutting, the 500 RFT Diamond Segmented Saw is ideal...",
  "recommended_products": [
    {
      "product_name": "500 RFT Diamond Segmented Saw",
      "brand": "Cumi 500 RFT",
      "badges": ["Cutting Blades", "Diamond Saw"],
      "key_specs": { "Dia (D)": "110 mm", "Thickness (T)": "1.8 mm" },
      "action_links": {
        "store_url": "https://cumiabrasive.in/product/...",
        "catalogue_pdf": "https://cumiabrasive.in/..."
      }
    }
  ],
  "suggested_follow_ups": [
    "Do you have this in 125mm?",
    "What is the price?",
    "Can it cut marble as well?"
  ]
}
```

Interactive docs available at `http://localhost:8000/docs`.
