from fastapi import FastAPI
from pydantic import BaseModel
from app.rag_pipeline import RAGPipeline
from app.logger import setup_logging
import logging

setup_logging()
logger = logging.getLogger("rag-api")

app = FastAPI()
rag = RAGPipeline()

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask_question(request: QuestionRequest):
    logger.info(f"Received question: {request.question}")

    result = rag.ask(request.question)

    logger.info("Returning response")

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": result["sources"],
        "model": rag.model_name
    }
