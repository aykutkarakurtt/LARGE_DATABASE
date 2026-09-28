from embedder import Embedder
from vector_store import VectorStore


class SearchEngine:
    def __init__(self):
        self.embedder = Embedder()
        dimension = getattr(
            self.embedder.model,
            "get_embedding_dimension",
            None,
        )
        if dimension is None:
            dimension = self.embedder.model.get_sentence_embedding_dimension()
        self.vector_store = VectorStore(dimension=dimension())

    def ingest(self, chunks, save=True):
        if not chunks:
            return 0
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedder.encode(texts)
        self.vector_store.add(embeddings, chunks)
        if save:
            self.vector_store.save()
        return len(chunks)

    def search(self, query, top_k=5, threshold=0.0):
        if not isinstance(query, str) or not query.strip():
            return []
        if self.vector_store.is_empty():
            return []

        top_k = max(1, int(top_k))
        query_embedding = self.embedder.encode_query(query.strip())
        return self.vector_store.search(
            query_embedding,
            top_k=top_k,
            threshold=float(threshold),
        )
