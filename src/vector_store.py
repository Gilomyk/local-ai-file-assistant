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

    def search(
            self,
            query_embedding,
            top_k: int = 5,
            source_filter: str | None = None,
    ):
        search_k = top_k

        if source_filter is not None:
            search_k = self.index.ntotal

        scores, indices = self.index.search(
            query_embedding,
            search_k,
        )

        results = []

        for score, index in zip(scores[0], indices[0]):
            if index == -1:
                continue

            document = self.documents[index]

            if source_filter is not None:
                if document["source"] != source_filter:
                    continue

            results.append(
                {
                    "score": float(score),
                    "source": document["source"],
                    "text": document["text"],
                }
            )

            if len(results) >= top_k:
                break

        return results