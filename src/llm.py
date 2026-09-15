import requests

from .config import OLLAMA_MODEL, OLLAMA_URL


def generate_answer(context: str, question: str) -> str:

    prompt = f"""
You are a document question-answering assistant.

Answer the user's question using ONLY information explicitly stated
in the provided context.

Rules:
- Do not use outside knowledge.
- Do not make assumptions or logical inferences.
- Do not add information that is not explicitly present in the context.
- If the answer cannot be found in the context, say:
  "I don't know based on the provided documents."
- Answer in the same language as the user's question.

Context:
{context}

Question:
{question}

Answer:
"""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    return response.json()["response"]