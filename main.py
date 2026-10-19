import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.rag_pipeline import ask_policy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="PolicyMind AI",
    description="API untuk tanya jawab kebijakan perusahaan.",
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1,
        max_length=1000,
        description="Pertanyaan tentang kebijakan perusahaan.",
    )


@app.get("/health")
def health_check():
    return {"status": "ok", "service": "PolicyMind AI"}


@app.post("/ask")
def ask(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=422,
            detail="Pertanyaan tidak boleh kosong.",
        )

    try:
        return ask_policy(question)
    except Exception:
        logger.exception("Gagal memproses pertanyaan pada endpoint /ask")
        raise HTTPException(
            status_code=500,
            detail="Terjadi kesalahan saat memproses pertanyaan.",
    )  
