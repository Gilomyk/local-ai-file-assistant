# Local AI File Assistant

A small local Retrieval-Augmented Generation (RAG) application for asking questions about user-provided documents.

The project uses **Sentence Transformers** for embeddings, **FAISS** for semantic retrieval and **Llama 3.2** running locally through **Ollama** to generate answers.

## Features

* PDF, DOCX and TXT document support
* Text chunking with configurable overlap
* Semantic search with Sentence Transformers + FAISS
* Local LLM inference with Llama 3.2
* Source-aware retrieval when a filename is specified
* Source names included in the retrieved context
* Configurable embedding model, chunk size, overlap, `TOP_K` and LLM

## How it works

```text
Documents
    ↓
Load & chunk
    ↓
Embeddings
    ↓
FAISS
    ↓
Relevant chunks
    ↓
Llama 3.2
    ↓
Answer
```

## Requirements

* Python 3.10+
* [Ollama](https://ollama.com/)
* An Ollama-compatible local LLM

## Setup

Clone the repository and create a virtual environment:

```powershell
git clone <REPOSITORY_URL>
cd local-ai-file-assistant

python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

Download the Llama 3.2 model:

```powershell
ollama pull llama3.2
```

## Usage

Place PDF, DOCX or TXT files in:

```text
data/documents/
```

Then run:

```powershell
python -m src.main
```

Example:

```text
Question (or 'exit'): What programming languages are mentioned in CV.pdf?

Answer:
Python, JavaScript, Java, C/C++.

Sources:
- CV.pdf (similarity: 0.296)
- CV.pdf (similarity: 0.292)
...
```

Type `exit` to close the application.

When a filename is explicitly mentioned in the question, retrieval is restricted to that document. Otherwise, semantic search is performed across all loaded documents.

## Configuration

Main settings are available in `src/config.py`:

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

TOP_K = 5

OLLAMA_MODEL = "llama3.2"
```

## Limitations

* The FAISS index is rebuilt on every run.
* Documents are processed again on every startup.
* Scanned PDFs are not processed with OCR.
* Retrieval quality depends on the embedding model and chunking strategy.
* The LLM can still produce incorrect answers when the retrieved context is ambiguous or incomplete.

## Technologies

* Python
* Sentence Transformers
* FAISS
* Ollama
* Llama 3.2
* PyPDF
* python-docx
* NumPy
