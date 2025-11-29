# RAG-Based-Study-Assistant-
RAG-Based Intelligent Study Assistant for Document-Oriented Question Answering and Flashcard Generation


## Installation
* Install requirements
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ```

* To download gemma-3-1b-it LLM model, **setup.py** needs to be run initially (this step requires login to huggingface to get model locally, follow the error codes).

  ```bash
  python3 setup.py
  ```

* Then we can run the server

  ```bash
  uvicorn server.rag_server:app --reload --port 8000
  ```

Its an opensource project which mainly influenced the RAG pipeline by this YouTube video:

https://www.youtube.com/watch?v=qN_2fnOPY-M
