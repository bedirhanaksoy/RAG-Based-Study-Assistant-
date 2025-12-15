import random
import pandas as pd
from modules.pdf_reader import (
    open_and_read_pdf,
    split_sentences,
    create_sentence_chunks,
    create_page_chunks
)
from modules.embedder import (
    load_embedding_model,
    embed_chunks,
    save_embeddings,
    load_embeddings
)
from modules.llm_inference import load_llm
from modules.rag_pipeline import ask


PDF_PATH = "data/pdf/human.pdf"
EMB_PATH = "data/embeddings/chunks_meta.csv"
PDF_URL = "https://pressbooks.oer.hawaii.edu/humannutrition2/open/download?type=pdf"


def build():
    pages_and_texts = open_and_read_pdf(PDF_PATH)
    pages_and_texts = split_sentences(pages_and_texts)
    pages_and_texts = create_sentence_chunks(pages_and_texts)

    # Returns list of dicts containing page_number + sentence_chunk + stats
    pages_and_chunks_over_min_len = create_page_chunks(pages_and_texts)

    # Prepare text chunks for embedding
    text_chunks = [item["sentence_chunk"] for item in pages_and_chunks_over_min_len]

    # Embed
    embedding_model = load_embedding_model()
    embeddings_tensor = embed_chunks(text_chunks, embedding_model)

    # Build DF EXACTLY like original script
    df = pd.DataFrame(pages_and_chunks_over_min_len)

    # Add embeddings as list
    df["embedding"] = embeddings_tensor.tolist()

    # Save EXACTLY like original code
    save_embeddings(df, EMB_PATH)

    print("[OK] Build complete.")


def run_example():
    pages_and_chunks, embeddings_tensor = load_embeddings(EMB_PATH)
    embedding_model = load_embedding_model()
    tokenizer, llm_model = load_llm()

    queries = [
        "What are the macronutrients, and what roles do they play in the human body?",
        "How often should infants be breastfed?",
        "What are symptoms of pellagra?"
    ]

    query = random.choice(queries)
    print(f"Query: {query}")

    answer, ctx = ask(
        query, tokenizer, llm_model,
        embedding_model, embeddings_tensor,
        pages_and_chunks, return_answer_only=False
    )

    print("Answer:")
    print(answer)
    print("Context:")
    print(ctx)


if __name__ == "__main__":
    try:
        run_example()
    except FileNotFoundError:
        build()
        run_example()
