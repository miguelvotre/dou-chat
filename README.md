# DOU Chat

![DOU Chat](assets/screenshot.png)

**Semantic search and Q&A over Brazil's Diário Oficial da União using RAG.**

Legal professionals spend hours manually scanning the DOU for relevant acts. DOU Chat lets them ask questions in natural language and get cited, source-grounded answers in seconds.

---

## How it works

```
User question
     │
     ▼
Embedding (multilingual-e5-large)
     │
     ▼
Vector similarity search (DuckDB VSS / MotherDuck)
     │
     ▼
Top-k chunks retrieved with metadata (órgão, seção, data)
     │
     ▼
LLM generation with grounded context (Llama 3.3 via Groq / Ollama)
     │
     ▼
Answer with source citations
```

---

## Features

- Natural language Q&A over DOU publications (Seções 1, 2, 3 and extras)
- Filters by section, issuing body, and date range
- Source citations on every answer (órgão, seção, date)
- API fallback chain on rate limits: Gemini → Qwen3 → Llama 3.3 → Magistral _(todo)_
- Local inference support via Ollama (phi4:14b)
- Bilingual UI (English / Portuguese) _(todo)_
- Evaluation pipeline with TruLens (answer relevance, context relevance, groundedness) _(todo)_

---

## Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| Embeddings | `intfloat/multilingual-e5-large`, LaBSE, BAAI/bge-m3 |
| Vector store | DuckDB VSS / MotherDuck |
| LLM (API) | Gemini 2.5 Flash, Qwen3 32B, Llama 3.3 70B, Magistral Medium |
| LLM (local) | phi4:14b via Ollama |
| LLM router | LiteLLM |
| Evaluation | TruLens |
| Observability | Arize Phoenix _(todo)_ |
| Ingestion | Python + schedule (Airflow planned) |

---

## Project structure

```
├── app.py                  # Streamlit chatbot
├── ingestion/              # DOU download and parsing
├── indexing/               # Chunking, embedding, and MotherDuck storage
├── rag/                    # Query engine (retrieval + generation)
├── evaluation/             # TruLens test queries
├── run_trulens_eval.py     # Evaluation runner
└── observability/          # Phoenix tracing setup
```

---

## Setup

```bash
# 1. Clone and install
python -m venv .venv && source .venv/bin/activate
pip install -e .

# 2. Set environment variables
cp .env.example .env
# Add MOTHERDUCK_TOKEN, GEMINI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY

# 3. Run the chatbot
streamlit run app.py
```

For local inference, start Ollama with `phi4:14b` before running the app.

---

## Evaluation _(todo)_

The evaluation pipeline uses TruLens with LLM-as-judge to score:
- **Answer relevance** — does the answer address the question?
- **Context relevance** — are the retrieved chunks relevant?
- **Groundedness** — is the answer supported by the retrieved context?

```bash
python run_trulens_eval.py --mode api
```

Results are stored locally and can be inspected in the TruLens dashboard.
