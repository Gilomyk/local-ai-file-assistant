from pathlib import Path
from typing import Sized

import numpy as np

from .chunker import split_text
from .config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DOCUMENTS_DIR,
    TOP_K,
)
from .embeddings import EmbeddingModel
from .llm import generate_answer
from .loaders import load_document
from .vector_store import VectorStore


def load_all_documents():
    documents = []

    for path in DOCUMENTS_DIR.iterdir():

        if not path.is_file():
            continue

        try:
            text = load_document(path)
        except ValueError:
            print(f"Skipping unsupported file: {path.name}")
            continue

        chunks = split_text(
            text,
            CHUNK_SIZE,
            CHUNK_OVERLAP,
        )

        for chunk in chunks:
            documents.append(
                {
                    "source": path.name,
                    "text": chunk,
                }
            )

    return documents

def find_source_filter(question: str, documents: list[dict]) -> Sized | None:
    question_lower = question.lower()

    sources = {document["source"] for document in documents}

    matching_sources = [
        source
        for source in sources
        if source.lower() in question_lower
    ]

    if not matching_sources:
        return None

    return max(matching_sources, key=len)

def main():

    print("Loading documents...")

    documents = load_all_documents()

    if not documents:
        print("No documents found.")
        print(f"Put PDF, DOCX or TXT files into: {DOCUMENTS_DIR}")
        return

    print(f"Loaded {len(documents)} chunks.")

    embedding_model = EmbeddingModel()

    texts = [document["text"] for document in documents]

    embeddings = embedding_model.encode(texts)

    embeddings = np.asarray(
        embeddings,
        dtype="float32",
    )

    vector_store = VectorStore(
        embeddings.shape[1],
    )

    vector_store.add(
        embeddings,
        documents,
    )

    print("Ready.")

    while True:

        question = input("\nQuestion (or 'exit'): ")

        if question.lower() == "exit":
            break

        query_embedding = embedding_model.encode([question])
        query_embedding = np.asarray(query_embedding, dtype="float32")

        source_filter = find_source_filter(question, documents)

        results = vector_store.search(
            query_embedding,
            TOP_K,
            source_filter=source_filter,
        )

        context = "\n\n".join(
            f"[Source: {result['source']}]\n{result['text']}"
            for result in results
        )

        answer = generate_answer(
            context,
            question,
        )

        print("\nAnswer:")
        print(answer)

        print("\nSources:")

        for result in results:
            print(
                f"- {result['source']} "
                f"(similarity: {result['score']:.3f})"
            )


if __name__ == "__main__":
    main()