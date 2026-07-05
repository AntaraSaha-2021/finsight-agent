# FinSight Agent 🔍

> An agentic AI system for financial research — autonomously retrieves, 
> reasons over, and synthesizes information from financial documents 
> and live web sources.

## Status
🚀 *Deployment coming soon — HuggingFace Spaces*

---

## What This Is

FinSight is a **genuinely agentic** financial research assistant. Unlike a standard RAG pipeline that retrieves and answers in one fixed step, FinSight uses a **ReAct loop** (Reason + Act) — the agent decides which tools to use, observes the results, and re-reasons before producing a final answer.

A user can ask:
> *"Summarize the key risks in this annual report and compare with the latest news about the company"*

The agent will:
1. Read the question and decide to search uploaded documents first
2. Retrieve relevant chunks from the FAISS index
3. Re-evaluate — decide web search adds value
4. Search the web for live news
5. Synthesize both sources into a cited, grounded answer

---

## Architecture

```
User Question
      │
      ▼
┌─────────────┐
│  Supervisor │  ← LangGraph node — reasons about what to do next
│    Node     │
└──────┬──────┘
       │ conditional edge (ReAct loop)
  ┌────┴─────┐
  ▼          ▼
[RAG]    [Web Search]
  │          │
  └────┬─────┘
       │ returns to supervisor
       ▼
┌─────────────┐
│   Answer    │  ← synthesizes all gathered context
│    Node     │
└─────────────┘
       │
       ▼
  FastAPI REST
  + HTML Frontend
```

**Key architectural decisions:**
- **ReAct pattern** — supervisor re-evaluates after every tool call, no hardcoded routing
- **Factory pattern** — LLM provider abstracted behind `get_llm()`, swap Groq ↔ Ollama via `.env`
- **Tool isolation** — RAG and web search are independent, stateless tools
- **Memory layer** — session history injected into every agent call, capped at 10 turns
- **Auth** — Bearer token on all protected endpoints, validated before agent runs
- **Input sanitization** — prompt injection guard at Pydantic schema level

---

## Stack
| Layer | Technology | Why |
|---|---|---|
| **Orchestration** | LangGraph 1.2.7 | Production-grade state machine, used at enterprise scale |
| **LLM** | Groq — Llama 3.1 8B | Free inference API, 560 T/sec, 131K context window |
| **LLM (local dev)** | Ollama + Mistral | No internet needed for development iteration |
| **RAG** | LangChain + FAISS | Persistent vector index, proven combination |
| **Embeddings** | all-MiniLM-L6-v2 | Free, local, no API key required |
| **Web Search** | ddgs (DuckDuckGo) | Free, no API key required |
| **Memory** | In-memory session store | Provider-agnostic interface, swappable to Redis |
| **API** | FastAPI + Pydantic | Type-safe endpoints, automatic OpenAPI docs |
| **Frontend** | Vanilla HTML/JS | No build step, served directly from FastAPI |


## Setup
*(full instructions coming soon)*

## Features
- [x] ReAct agentic loop — supervisor re-reasons after every tool call
- [x] Financial PDF ingestion with validation (size, type, empty content guards)
- [x] FAISS vector store — persistent, multi-document index
- [x] Live web search with retry and graceful degradation
- [x] Multi-turn conversation memory with session isolation
- [x] FastAPI REST API with Bearer token authentication
- [x] Prompt injection guard on all user inputs
- [x] Provider-agnostic LLM via Factory Pattern (Groq / Ollama)
- [x] CORS configured via environment variables
- [x] Single-page chat frontend — upload, chat, session management
- [x] 43 tests across all layers (config, LLM, API, graph, tools, memory)

---
## Project Structure

```
finsight-agent/
├── agents/
│   ├── graph.py           # LangGraph StateGraph — ReAct loop
│   ├── nodes.py           # supervisor, rag, web_search, answer nodes
│   ├── state.py           # AgentState TypedDict
│   └── llm_factory.py     # Factory Pattern — Groq or Ollama
├── api/
│   ├── main.py            # FastAPI app — all endpoints
│   └── schemas.py         # Pydantic models with injection guard
├── config/
│   └── settings.py        # Single source of all configuration
├── frontend/
│   └── index.html         # Single-page chat UI
├── memory/
│   └── session_memory.py  # Session memory — swappable backend
├── tools/
│   ├── rag_tool.py        # PDF ingestion + FAISS retrieval
│   └── web_search_tool.py # DuckDuckGo search with retry
├── tests/                 # 43 tests — pytest
├── Dockerfile             # HuggingFace Spaces deployment
├── app.py                 # HF Spaces entry point
└── requirements.txt       # Pinned versions
```

---

## Running Locally

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.ai) installed (for local LLM)
- [Groq API key](https://console.groq.com) (free — for deployment LLM)

### Setup

**1. Clone and create virtual environment:**
```bash
git clone https://github.com/AntaraSaha-2021/finsight-agent.git
cd finsight-agent
python -m venv .venv

# Activate:
# Windows:  .venv\Scripts\activate
# Mac/Linux: source .venv/bin/activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

**3. Configure environment:**
```bash
cp .env.example .env
# Edit .env and fill in your values
```

Required `.env` values:
```bash
GROQ_API_KEY=your_groq_key_here        # from console.groq.com
API_SECRET_KEY=your_generated_secret   # run: python -c "import secrets; print(secrets.token_hex(32))"
LLM_PROVIDER=groq                      # or 'ollama' for local
```

**4. Pull Ollama model (if using local LLM):**
```bash
ollama pull mistral
```

**5. Run — requires two terminals:**

*Terminal 1 — API server:*
```bash
uvicorn api.main:app --reload
```

*Terminal 2 — Frontend:*
```
Open frontend/index.html with VS Code Live Server
OR visit http://127.0.0.1:8000 (served by FastAPI directly)
```

**6. Run tests:**
```bash
pytest tests/ -v
```

Expected: 43 tests passing.

---

## API Reference

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| GET | `/health` | None | Liveness check — LLM provider, index status, active sessions |
| POST | `/chat` | Bearer | Send question, get agent answer with sources |
| POST | `/upload` | Bearer | Upload PDF for RAG indexing |
| DELETE | `/session/{id}` | Bearer | Clear conversation memory for a session |
| GET | `/docs` | None | Interactive API explorer (Swagger UI) |

### Example Request

```bash
curl -X POST http://localhost:8000/chat \
  -H "Authorization: Bearer YOUR_API_SECRET_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total revenue?", "session_id": "my-session"}'
```

---

## Security Decisions

| Decision | Implementation |
|---|---|
| API authentication | HTTP Bearer token on all protected endpoints |
| Prompt injection guard | Regex pattern matching at Pydantic schema level |
| CORS | Explicit allowed origins via `ALLOWED_ORIGINS` env var |
| Secrets | Never hardcoded — all via `.env` / environment variables |
| PDF validation | Size cap (50MB), type check, empty content detection |
| Session isolation | Session IDs validated (alphanumeric only), no path traversal |
| `data/` and `vectorstore/` | Gitignored — sensitive financial documents never committed |

---

## Design Patterns Used

- **ReAct (Reason + Act)** — agent loop with supervisor re-evaluation after each tool
- **Factory Pattern** — `get_llm()` abstracts LLM provider, zero agent code changes to swap
- **Strategy Pattern** — session memory backend swappable (in-memory → Redis → ChromaDB)
- **Separation of Concerns** — tools, agents, memory, and API are fully decoupled layers
- **Fail Loud at Startup** — graph compiles and LLM connects at app start, not mid-request

---

## Known Limitations

| Limitation | Notes |
|---|---|
| FAISS index resets on container restart | Free tier behaviour. Production: use managed vector DB |
| Uploaded PDFs are ephemeral | Reset on container restart. Production: use object storage |
| Session memory is in-process | Resets on server restart. Production: use Redis |
| Text-based PDFs only | Scanned/image PDFs have no extractable text |
| DuckDuckGo is unofficial | May rate limit. Retry logic included |

---
*Built by [Antara Saha](https://github.com/AntaraSaha-2021) · July 2026*  
*Python · LangGraph · FastAPI · Groq · FAISS*