# Cloud Native RAG API

Lightweight FastAPI RAG service using:

- Vertex AI Gemini (`gemini-2.5-flash-lite`)
- Vertex AI Embeddings (`text-embedding-005`)
- FAISS vector search
- Docker

This is a learning/demo project, not a hardened production service.

---

## What It Does

1. Accepts a question via `POST /ask`
2. Embeds the question with Vertex embeddings
3. Retrieves top-k chunks from FAISS
4. Prompts Gemini with retrieved context
5. Returns answer + source chunks

---

## Important Requirement (GCP)

The API depends on Vertex AI for both generation and embeddings.  
Docker alone is not enough; valid Google credentials are required for `/ask` to work.

Minimum setup:

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

Note: code currently uses `us-central1`.

---

## Quickstart (Docker First)

### 1. Build image

```bash
docker build -t cloud-native-rag-api .
```

### 2. Run container

```bash
export PROJECT_ID="YOUR_PROJECT_ID"
docker run --rm -p 8080:8080 \
  -e GOOGLE_CLOUD_PROJECT="$PROJECT_ID" \
  -e GOOGLE_APPLICATION_CREDENTIALS=/root/.config/gcloud/application_default_credentials.json \
  -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
  cloud-native-rag-api
```

Open docs:

```text
http://localhost:8080/docs
```

### 3. Test endpoint

```bash
curl -X POST "http://localhost:8080/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"What is Retrieval-Augmented Generation?"}'
```

Expected response shape:

```json
{
  "question": "...",
  "answer": "...",
  "sources": ["...", "...", "..."],
  "model": "gemini-2.5-flash-lite"
}
```

---

## Local Development (Without Docker)

### 1. Create and activate venv

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Authenticate to GCP

```bash
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID
```

Optional explicit env vars:

```bash
export GOOGLE_CLOUD_PROJECT=YOUR_PROJECT_ID
export VERTEX_LOCATION=us-central1
```

### 4. Run API

```bash
uvicorn app.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

---

## Security Notes (Minimal)

- Do not commit credentials or service-account key files.
- `--allow-unauthenticated` is fine for learning demos, but avoid it for shared/public use.
- Set a small budget alert to avoid accidental GCP spend.

---

## Cloud Run Deployment (Demo)

```bash
gcloud run deploy cloud-native-rag-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

For safer deployment, remove `--allow-unauthenticated` and require IAM-authenticated callers.

---

## Current Limitations

- Knowledge base is currently in-memory sample text (not external corpus ingestion).
- FAISS index is created at startup (no persistent managed vector DB).
- No auth/rate limiting at API layer yet.