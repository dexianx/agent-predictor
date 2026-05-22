# Starting the Kora Task A Servers

Two processes must be running: the **FastAPI backend** (port 8001) and the **Vite frontend** (port 8080).

---

## Prerequisites

### Python dependencies (one-time)
```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### Node dependencies (one-time)
```bash
cd ui && bun install
```

### Environment variables
Copy `.env.example` to `.env` in the project root and fill in:
```
# LLM: gemini (default) | groq | cerebras | mistral
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-2.5-flash

# For Groq instead:
# LLM_PROVIDER=groq
# GROQ_API_KEY=your_groq_key
# GROQ_MODEL=llama-3.3-70b-versatile

# For Cerebras (OpenAI-compatible — https://inference-docs.cerebras.ai/resources/openai):
# LLM_PROVIDER=cerebras
# CEREBRAS_API_KEY=your_cerebras_key
# CEREBRAS_MODEL=gpt-oss-120b

# For Mistral (OpenAI-compatible — https://docs.mistral.ai/api):
# LLM_PROVIDER=mistral
# MISTRAL_API_KEY=your_mistral_key
# MISTRAL_MODEL=mistral-small-2506

# Optional cross-provider override: LLM_MODEL=custom-model-id

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password
SQLITE_PATH=./personas.db
```

The frontend reads `ui/.env.local` — it is already configured:
```
VITE_API_URL=http://localhost:8001
```

---

## Starting the backend

From the **project root**:
```bash
.venv/bin/uvicorn api.main:app --reload --port 8001
```

The API will be available at `http://localhost:8001`.  
Hot-reload watches the entire project directory.

Verify it is healthy:
```bash
curl http://localhost:8001/health
# → {"status":"ok"}
```

---

## Starting the frontend

From the **`ui/` directory**:
```bash
cd ui
bun run dev
```

The UI will be available at `http://localhost:8080`.

---

## Running both together (two terminals)

**Terminal 1 — backend:**
```bash
.venv/bin/uvicorn api.main:app --reload --port 8001
```

**Terminal 2 — frontend:**
```bash
cd ui && bun run dev
```

---

## Running tests

Requires the backend to be running for the WebSocket test in `test_api.py`.

```bash
# Unit tests only (no server needed)
.venv/bin/pytest tests/test_llm.py tests/test_pipeline.py -v

# Full suite (start backend first)
.venv/bin/pytest tests/ -v
```
