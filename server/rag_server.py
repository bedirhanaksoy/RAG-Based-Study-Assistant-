from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

from modules.embedder import load_embeddings, load_embedding_model
from modules.llm_inference import load_llm
from modules.rag_pipeline import ask


# ============================================================
# Load EVERYTHING exactly like the original main.py
# ============================================================

EMB_PATH = "data/embeddings/chunks.csv"


print("[SERVER] Loading embeddings...")
pages_and_chunks, embeddings_tensor = load_embeddings(EMB_PATH)

print("[SERVER] Loading embedding model (SentenceTransformer)...")
embedding_model = load_embedding_model()

print("[SERVER] Loading LLM (Gemma 3.1B IT)...")
tokenizer, llm_model = load_llm()

print("[SERVER] All components loaded successfully.")

# ============================================================
# FastAPI setup
# ============================================================

app = FastAPI(title="Local RAG Server", version="1.0")


class QueryRequest(BaseModel):
    query: str
    temperature: float = 0.7
    max_new_tokens: int = 512
    return_context: bool = False


@app.post("/ask")
@app.post("/ask")
def ask_rag(request: QueryRequest):
    # EXACT same pipeline call
    output = ask(
        query=request.query,
        tokenizer=tokenizer,
        llm_model=llm_model,
        embedding_model=embedding_model,
        embeddings=embeddings_tensor,
        pages_and_chunks=pages_and_chunks,
        temperature=request.temperature,
        max_new_tokens=request.max_new_tokens,
        return_answer_only=not request.return_context
    )

    # ------------------------
    # CASE 1: Return only answer
    # ------------------------
    if request.return_context is False:
        return {"answer": output}

    # ------------------------
    # CASE 2: Return answer + SAFE context
    # ------------------------
    answer, context_items = output

    # CLEAN context for JSON (remove embeddings, scores, numpy etc.)
    cleaned_context = [
        {
            "page_number": item["page_number"],
            "sentence_chunk": item["sentence_chunk"]
        }
        for item in context_items
    ]

    return {
        "answer": answer,
        "context": cleaned_context
    }


@app.get("/")
def root():
    return {"message": "Local RAG Server is running."}
