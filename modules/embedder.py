import os
import json
import torch
import numpy as np
import pandas as pd
import ast
from sentence_transformers import SentenceTransformer, util

try:
    import faiss
except Exception:
    faiss = None


def load_embedding_model():
    return SentenceTransformer("all-mpnet-base-v2", device="cuda")


def embed_chunks(text_chunks, model):
    return model.encode(
        text_chunks,
        batch_size=16,
        convert_to_tensor=True
    )


def save_embeddings(df, path):
    """Persist embeddings using a FAISS index + metadata file.

    Kept signature compatible with previous implementation which wrote a CSV.
    This will create two files sharing the given path's basename:
      - <base>.index    (FAISS index with vectors)
      - <base>_meta.json (metadata as JSON, handles all special characters)
    """
    base = os.path.splitext(path)[0]
    meta_path = f"{base}_meta.json"
    index_path = f"{base}.index"

    # Extract metadata (everything except the embedding column)
    if "embedding" in df.columns:
        meta_df = df.drop(columns=["embedding"]) 
    else:
        meta_df = df.copy()

    # Save metadata as JSON (handles all special characters natively)
    meta_list = meta_df.to_dict(orient="records")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)

    # Prepare embeddings array
    if "embedding" in df.columns:
        emb_array = np.array(df["embedding"].tolist(), dtype=np.float32)
    else:
        # Nothing to save for embeddings
        emb_array = None

    # Write FAISS index
    if emb_array is not None:
        if faiss is None:
            raise RuntimeError("faiss is required to save embeddings as an index. Install faiss or revert to CSV storage.")

        dim = emb_array.shape[1]
        # Use Inner Product index to match dot_score usage
        index = faiss.IndexFlatIP(dim)
        index.add(emb_array)
        faiss.write_index(index, index_path)


def load_embeddings(path):
    base = os.path.splitext(path)[0]
    meta_path = f"{base}_meta.json"
    index_path = f"{base}.index"

    # If a FAISS index + JSON meta exists, load from them
    if os.path.exists(index_path) and os.path.exists(meta_path):
        if faiss is None:
            raise RuntimeError("faiss is required to load FAISS-backed embeddings. Install faiss or provide CSV embeddings.")

        # Load metadata from JSON
        with open(meta_path, "r", encoding="utf-8") as f:
            pages_and_chunks = json.load(f)

        # Load FAISS index and reconstruct vectors
        index = faiss.read_index(index_path)
        ntotal = index.ntotal
        if ntotal == 0:
            embeddings = torch.empty((0, 0), dtype=torch.float32).to("cuda")
            return pages_and_chunks, embeddings

        # Reconstruct all vectors (IndexFlat supports reconstruct)
        emb_list = []
        for i in range(ntotal):
            vec = index.reconstruct(i)
            emb_list.append(np.array(vec, dtype=np.float32))

        embeddings = torch.tensor(np.array(emb_list, dtype=np.float32), dtype=torch.float32).to("cuda")
        return pages_and_chunks, embeddings

    # Fallback: legacy CSV format (keeps backward compatibility)
    df = pd.read_csv(path, quoting=1)
    df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
    pages_and_chunks = df.to_dict(orient="records")

    embeddings = torch.tensor(
        np.array(df["embedding"].tolist()),
        dtype=torch.float32
    ).to("cuda")

    return pages_and_chunks, embeddings


def retrieve_relevant_resources(query, embeddings, model, n_resources_to_return=5, print_time=True):
    query_embedding = model.encode(query, convert_to_tensor=True)

    dot_scores = util.dot_score(query_embedding, embeddings)[0]

    scores, indices = torch.topk(dot_scores, k=n_resources_to_return)

    return scores, indices
