# Cloud Native RAG API

A production-ready Retrieval-Augmented Generation (RAG) API built with:

- Vertex AI (Gemini 2.5 Flash-Lite)
- Vertex AI Embeddings (text-embedding-005)
- FAISS vector search
- FastAPI
- Docker
- Cloud Run ready

---

## Architecture

1. User sends question
2. Question is embedded using Vertex embeddings
3. FAISS performs similarity search
4. Retrieved context is injected into prompt
5. Gemini generates grounded response
6. API returns answer + sources

---

## Local Development

### 1. Create virtual environment

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
```

### 4. Run locally

```bash
uvicorn app.main:app --reload
```

Open:
```bash
http://127.0.0.1:8000/docs
```

## Docker
Build

```bash
docker build -t cloud-native-rag-api .
```

Run

```bash
docker run -p 8080:8080 cloud-native-rag-api
```

Open:

```bash
http://localhost:8080/docs
```

## Cloud Run Deployment

```bash
gcloud run deploy cloud-native-rag-api \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
```

Example Request
POST /ask

```bash
{
  "question": "What is Retrieval-Augmented Generation?"
}
```

### Tech Stack

FastAPI

- LangChain (modular v1 architecture)
- Vertex AI
- FAISS
- Docker
