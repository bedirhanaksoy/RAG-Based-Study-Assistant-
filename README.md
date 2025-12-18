# 📘 RAG-Based Study Assistant

**RAG-Based Intelligent Study Assistant for Document-Oriented Question Answering and Flashcard Generation**

This project is a **fully local, end-to-end Retrieval-Augmented Generation (RAG)** application designed to help users study large PDF documents efficiently through a built-in web interface.

The system allows users to upload textbooks or lecture notes, ask document-grounded questions, and generate flashcards,  all while running entirely on a local machine.

---

## 🚀 Key Features

- 📚 Upload and manage multiple PDF documents
- ❓ Ask questions grounded in the selected document
- 🧠 Generate flashcards using the same RAG pipeline
- 📄 Page-level source attribution for answers
- ⚡ Persistent FAISS vector indexes (no recomputation)
- 🔒 No external APIs, no cloud dependency

---
## 🧠 RAG Pipeline

```

PDF  
→ Text Cleaning & Chunking  
→ Embedding Model  
→ FAISS Vector Index  
→ Retriever  
→ Local LLM (Gemma)  
→ Answer / Flashcards + Sources  

````

The same retrieval pipeline is reused for:
- Question answering
- Flashcard generation

---

## 🛠️ Installation

### 1️⃣ Create virtual environment and install requirements

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
````

### 2️⃣ Download local LLM model

To download the **Gemma 3-1b IT** model locally, run:

```bash
python3 setup.py
```

> ⚠️ This step requires logging in to Hugging Face to download the model weights.  
> Follow the error messages if authentication is required.

---

## ▶️ Run the Server

```bash
uvicorn server.rag_server:app --reload --port 8000
```

Then open:

```
http://localhost:3000
```

You can now:

- Upload a PDF from the UI
    
- Select a document
    
- Ask questions
    
- Generate flashcards
    

---

## 🧠 Core Technologies

- **LLM:** Google Gemma (instruction-tuned, ~1B parameters)
    
- **Embeddings:** Sentence-level dense embeddings
    
- **Vector Store:** FAISS (local, persistent)
    
- **Backend:** FastAPI
    
- **Frontend:** Next.js
    
- **Storage:**
    
    - `.index` files for FAISS embeddings
        
    - `_meta.csv` files for text chunks & page numbers
        

---

## ⚠️ Limitations

    
- First-time embedding generation may be slow for large PDFs
    
- Retrieval quality is not yet quantitatively evaluated
    

---

## 📚 Acknowledgements

This project’s RAG pipeline design was mainly influenced by the following tutorial:

- **Daniel Bourke – Local Retrieval Augmented Generation (RAG) from Scratch**  
    [https://www.youtube.com/watch?v=qN_2fnOPY-M](https://www.youtube.com/watch?v=qN_2fnOPY-M)
    

---

## 👥 Authors

- **Bedirhan Ömer Aksoy**
- **Ahmet Semih Marufoğlu**
