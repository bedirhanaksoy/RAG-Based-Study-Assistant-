import torch
import numpy as np
import pandas as pd
import ast
from sentence_transformers import SentenceTransformer, util


def load_embedding_model():
    """Exact same model loading logic."""
    return SentenceTransformer("all-mpnet-base-v2", device="cuda")


def embed_chunks(text_chunks, model):
    """Exact same encode() parameters."""
    return model.encode(
        text_chunks,
        batch_size=16,
        convert_to_tensor=True
    )


def save_embeddings(df, path):
    df.to_csv(path, index=False)


def load_embeddings(path):
    df = pd.read_csv(path)
    df["embedding"] = df["embedding"].apply(lambda x: np.array(ast.literal_eval(x)))
    pages_and_chunks = df.to_dict(orient="records")

    embeddings = torch.tensor(
        np.array(df["embedding"].tolist()),
        dtype=torch.float32
    ).to("cuda")

    return pages_and_chunks, embeddings


def retrieve_relevant_resources(query, embeddings, model, n_resources_to_return=5, print_time=True):
    """100% identical behavior."""
    query_embedding = model.encode(query, convert_to_tensor=True)

    dot_scores = util.dot_score(query_embedding, embeddings)[0]

    scores, indices = torch.topk(dot_scores, k=n_resources_to_return)

    return scores, indices
