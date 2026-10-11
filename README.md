# LangGraph Intent Classifier Demo

A standalone learning project that classifies a query into only four outcomes:

- `sql_agent` — stored data, lists, counts, filters or grouping
- `compliance` — overall compliance health, risks and alerts
- `rag` — policy, rules, guidance and explanations
- `clarify` — insufficient confidence

It deliberately does not execute SQL or RAG. It only demonstrates classification.

## Workflow

```text
START
  → semantic_router (Gemini embedding + Qdrant)
      → score >= 0.70 → finalize
      → score < 0.70 or error → llm_router
          → confidence >= 0.70 → finalize
          → confidence < 0.70 or error → keyword_router
              → safe multi-signal match → finalize
              → no match → clarify → finalize
  → END
```

The keyword fallback requires at least two signals. One word such as `count` or `visa` cannot decide an intent.

## Configure

```bash
cp .env.example .env
```

Set at minimum:

```env
GEMINI_API_KEY=your-key
GROQ_API_KEY=your-key
```

The demo reuses the e-Migrant defaults:

- Embeddings: `gemini-embedding-001`, 3072 dimensions
- Primary router LLM: `openai/gpt-oss-120b` through Groq
- LLM fallback: Gemini

## Run with Docker

```bash
docker compose up --build
```

Open <http://localhost:8000>.

## Run locally

Start Qdrant:

```bash
docker compose up -d qdrant
```

Install and start the API:

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000> to use the frontend and API from the same origin.

You may also serve `frontend/` with VS Code Live Server on port `5500` or
`5501`. Those local origins are allowed through `CORS_ORIGINS`. Restart
FastAPI after changing `.env`.

## API example

```bash
curl -X POST http://localhost:8000/api/classify \
  -H "Content-Type: application/json" \
  -d '{"query":"What evidence is required for a right to work check?"}'
```

Useful endpoints:

- `POST /api/classify`
- `GET /api/intents`
- `POST /api/reseed`
- `GET /api/health`
- `GET /docs`

## Add training utterances

Edit `data/intents.json`, add representative sentences, then call:

```bash
curl -X POST http://localhost:8000/api/reseed
```

Add real spelling mistakes and paraphrases from logs. Keep the three intents conceptually separate.

## Understand the confidence

For semantic routing, Qdrant returns cosine similarity. The UI displays it as a percentage for readability, but it is not a probability that the classification is correct. Test the `0.70` threshold on labelled real queries before using it in production.

The LLM confidence is the model's self-assessment and is also not a calibrated probability.

## Run tests

```bash
pytest -q
```

The tests do not require API keys or a running Qdrant instance because external services are mocked.
