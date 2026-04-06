from fastapi import FastAPI
from fastapi import HTTPException
from pydantic import BaseModel
from app.rag_pipeline import RAGPipeline
from app.logger import setup_logging
import logging
from functools import lru_cache

setup_logging()
logger = logging.getLogger("rag-api")

app = FastAPI()

class QuestionRequest(BaseModel):
    question: str


@lru_cache
def get_rag() -> RAGPipeline:
    return RAGPipeline()


@app.post("/ask")
def ask_question(request: QuestionRequest):
    logger.info(f"Received question: {request.question}")

    try:
        rag = get_rag()
        result = rag.ask(request.question)
    except Exception as exc:
        logger.exception("RAG pipeline initialization/inference failed")
        raise HTTPException(
            status_code=500,
            detail=(
                "Vertex AI is not configured. "
                "Set Application Default Credentials and GOOGLE_CLOUD_PROJECT."
            ),
        ) from exc

    logger.info("Returning response")

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "model": rag.model_name
    }
