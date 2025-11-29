from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import os

from modules.embedder import (
    load_embeddings,
    load_embedding_model,
    embed_chunks,
    save_embeddings
)
from modules.llm_inference import load_llm
from modules.rag_pipeline import ask
from modules.pdf_reader import (
    open_and_read_pdf,
    split_sentences,
    create_sentence_chunks,
    create_page_chunks,
)

PDF_DIR = "data/pdf"
EMB_DIR = "data/embeddings"

print("[SERVER] Loading embedding model (SentenceTransformer)...")
embedding_model = load_embedding_model()

print("[SERVER] Loading LLM (Gemma 3.1B IT)...")
tokenizer, llm_model = load_llm()

print("[SERVER] Server components loaded (models). Embeddings will be loaded per-request.")

# ============================================================
# FastAPI setup
# ============================================================

app = FastAPI(title="Local RAG Server", version="1.0")


class QueryRequest(BaseModel):
    query: str
    pdf_name: Optional[str] = None
    temperature: float = 0.7
    max_new_tokens: int = 512
    return_context: bool = False


def _pdf_path_for_name(pdf_name: str) -> str:
    # Ensure .pdf extension
    if not pdf_name.lower().endswith(".pdf"):
        pdf_name = pdf_name + ".pdf"
    return os.path.join(PDF_DIR, pdf_name)


def _emb_path_for_pdf(pdf_name: str) -> str:
    # embeddings saved as <pdf_basename>.index and <pdf_basename>_meta.json
    base = os.path.splitext(os.path.basename(pdf_name))[0]
    return os.path.join(EMB_DIR, base)


def _ensure_embeddings_for_pdf(pdf_name: str):
    """Return (pages_and_chunks, embeddings_tensor).

    If embeddings file exists, load and return. If not, create embeddings from the PDF
    and save them, then load and return.
    Raises HTTPException(404) if the PDF file doesn't exist.
    """
    pdf_path = _pdf_path_for_name(pdf_name)
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail=f"PDF not found: {pdf_name}")

    emb_base = _emb_path_for_pdf(pdf_name)
    emb_index_path = f"{emb_base}.index"
    emb_meta_path = f"{emb_base}_meta.json"

    # Check if embeddings already exist (both index and metadata files)
    if os.path.exists(emb_index_path) and os.path.exists(emb_meta_path):
        # existing embeddings
        print(f"[SERVER] Loading existing embeddings for {pdf_name}...")
        return load_embeddings(emb_base)

    # create embeddings
    print(f"[SERVER] Creating embeddings for {pdf_name}...")
    pages_and_texts = open_and_read_pdf(pdf_path)
    pages_and_texts = split_sentences(pages_and_texts)
    pages_and_texts = create_sentence_chunks(pages_and_texts)
    pages_and_chunks_over_min_len = create_page_chunks(pages_and_texts)

    text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_len]

    embeddings_tensor = embed_chunks(text_chunks, embedding_model)

    df = pd.DataFrame(pages_and_chunks_over_min_len)
    df["embedding"] = embeddings_tensor.tolist()

    # Ensure embeddings dir exists
    os.makedirs(EMB_DIR, exist_ok=True)
    save_embeddings(df, emb_base)

    print(f"[SERVER] Embeddings created and saved to {emb_base}.*")

    return load_embeddings(emb_base)


@app.post("/ask")
def ask_rag(request: QueryRequest):
    # Determine which PDF/embeddings to use. If not provided, fall back to default
    pdf_name = request.pdf_name or "human.pdf"

    # Ensure embeddings exist (load or create)
    pages_and_chunks, embeddings_tensor = _ensure_embeddings_for_pdf(pdf_name)

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

    if request.return_context is False:
        return {"answer": output}

    answer, context_items = output

    cleaned_context = [
        {"page_number": item["page_number"], "sentence_chunk": item["sentence_chunk"]}
        for item in context_items
    ]

    return {"answer": answer, "context": cleaned_context}


@app.post("/upload")
def upload_pdf(file: UploadFile = File(...)):
    """Upload a PDF to data/pdf. Returns the saved filename.

    If a file with same name exists it will be overwritten.
    """
    filename = file.filename
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    os.makedirs(PDF_DIR, exist_ok=True)
    dest_path = os.path.join(PDF_DIR, filename)

    with open(dest_path, "wb") as f:
        content = file.file.read()
        f.write(content)

    return {"message": "uploaded", "filename": filename}


@app.get("/")
def root():
    return {"message": "Local RAG Server is running."}
