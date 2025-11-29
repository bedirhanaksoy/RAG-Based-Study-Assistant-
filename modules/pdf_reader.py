import os
import fitz
import re
import pandas as pd
from tqdm.auto import tqdm
from spacy.lang.en import English
import requests

nlp = English()
nlp.add_pipe("sentencizer")


def text_formatter(text: str) -> str:
    return text.replace("\n", " ").strip()


def open_and_read_pdf(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    for page_number, page in tqdm(enumerate(doc)):
        text = text_formatter(page.get_text())
        pages_and_texts.append({
            "page_number": page_number,
            "page_char_count": len(text),
            "page_word_count": len(text.split(" ")),
            "page_sentence_count_raw": len(text.split(". ")),
            "page_token_count": len(text) / 4,
            "text": text
        })
    return pages_and_texts


def split_sentences(pages_and_texts):
    for item in tqdm(pages_and_texts):
        item["sentences"] = [str(s) for s in nlp(item["text"]).sents]
        item["page_sentence_count_spacy"] = len(item["sentences"])
    return pages_and_texts


def split_list(input_list: list, slice_size: int):
    return [input_list[i:i + slice_size]
            for i in range(0, len(input_list), slice_size)]


def create_sentence_chunks(pages_and_texts, chunk_size=10):
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(item["sentences"], chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])
    return pages_and_texts


def create_page_chunks(pages_and_texts, min_token_length=30):
    pages_and_chunks = []

    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            joined = "".join(sentence_chunk).replace("  ", " ").strip()
            joined = re.sub(r'\.([A-Z])', r'. \1', joined)

            chunk_dict = {
                "page_number": item["page_number"] + 1,
                "sentence_chunk": joined,
                "chunk_char_count": len(joined),
                "chunk_word_count": len(joined.split(" ")),
                "chunk_token_count": len(joined) / 4
            }
            pages_and_chunks.append(chunk_dict)

    df = pd.DataFrame(pages_and_chunks)
    filtered = df[df["chunk_token_count"] > min_token_length].to_dict(orient="records")
    return filtered
