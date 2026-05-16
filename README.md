# SHL Assessment Recommender

Conversational agent that recommends SHL Individual Test Solutions via a FastAPI service.

## Architecture

```
POST /chat
   │
   ├─ TF-IDF Retrieval (top-15 candidates from catalog)
   │     + keyword domain boosts
   │
   ├─ Claude Sonnet 4 (LLM)
   │     system prompt injects retrieved candidates
   │     forced JSON output: {reply, recommendations[], end_of_conversation}
   │
   └─ Validation layer
         checks every recommendation name against full catalog
         drops hallucinated entries, canonicalises URLs
```

## Local Setup

```bash
# 1. Clone / create project dir
cd shl-recommender

# 2. Install deps
pip install -r requirements.txt

# 3. Set API key
export ANTHROPIC_API_KEY=sk-ant-...

# 4. Start server
uvicorn main:app --host 0.0.0.0 --port 8000

# 5. Health check
curl http://localhost:8000/health
# → {"status":"ok"}

# 6. Test chat
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Hiring a Java developer who works with stakeholders"}]}'
```

## Deploying to Render (Free Tier)

1. Push this directory to a GitHub repo.
2. Go to https://render.com → New → Web Service → connect your repo.
3. Choose **Docker** as runtime (Render detects `render.yaml` automatically).
4. Add environment variable: `ANTHROPIC_API_KEY = sk-ant-...`
5. Deploy. Your URL will be `https://shl-recommender-xxxx.onrender.com`.

> **Cold start**: Render free tier sleeps after 15 min inactivity. First `/health` call may take ~60 s.

## Running Evaluation

```bash
# Against local server
python eval.py --url http://localhost:8000

# Against deployed server
python eval.py --url https://your-app.onrender.com
```

## API Reference

### GET /health
```json
{"status": "ok"}
```

### POST /chat
**Request**
```json
{
  "messages": [
    {"role": "user", "content": "I need to hire a Java developer"},
    {"role": "assistant", "content": "...previous reply..."},
    {"role": "user", "content": "Mid-level, around 4 years experience"}
  ]
}
```

**Response**
```json
{
  "reply": "Here are 4 assessments that fit a mid-level Java developer...",
  "recommendations": [
    {"name": "Java 8 (New)", "url": "https://www.shl.com/...", "test_type": "K"},
    {"name": "OPQ32r",       "url": "https://www.shl.com/...", "test_type": "P"}
  ],
  "end_of_conversation": false
}
```

## Files

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, agent logic, request validation |
| `retrieval.py` | TF-IDF index + keyword boosting |
| `catalog_data.py` | SHL Individual Test Solutions catalog |
| `eval.py` | Evaluation harness (Recall@10 + behavior probes) |
| `Dockerfile` | Container definition for deployment |
| `render.yaml` | Render.com deployment config |
| `requirements.txt` | Python dependencies |
