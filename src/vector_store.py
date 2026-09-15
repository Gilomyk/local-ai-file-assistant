import faiss


class VectorStore:

    def __init__(self, dimension: int):
        self.index = faiss.IndexFlatIP(dimension)
        self.documents = []

    def add(self, embeddings, documents):
        self.index.add(embeddings)
        self.documents.extend(documents)

    def search(self, query_embedding, top_k: int = 5):
        scores, indices = self.index.search(
            query_embedding,
            top_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            document = self.documents[index]

            results.append(
                {
                    "score": float(score),
                    "source": document["source"],
                    "text": document["text"],
                }
            )

        return results