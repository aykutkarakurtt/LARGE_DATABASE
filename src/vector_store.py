import json
import os

import faiss
import numpy as np

from config import INDEX_FILE, METADATA_FILE, VECTORSTORE_DIR


class VectorStore:
    def __init__(self, dimension=768):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(dimension)
        self.metadata = []
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        if os.path.exists(INDEX_FILE) and os.path.exists(METADATA_FILE):
            self.load()

    def reset(self):
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata = []

    def add(self, embeddings, chunks):
        if len(embeddings) != len(chunks):
            raise ValueError("Her embedding için bir metin parçası gerekir")
        if len(embeddings) == 0:
            return

        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.ndim != 2 or embeddings.shape[1] != self.dimension:
            raise ValueError(
                "Beklenen embedding boyutu "
                f"{self.dimension}, alınan {embeddings.shape}"
            )
        if not np.isfinite(embeddings).all():
            raise ValueError("Embedding değerleri sonlu olmalı")

        self.index.add(embeddings)
        self.metadata.extend(chunks)

    def search(self, query_embedding, top_k=5, threshold=0.5):
        if self.is_empty():
            return []

        query_embedding = np.asarray(query_embedding, dtype=np.float32)
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        if query_embedding.shape != (1, self.dimension):
            raise ValueError(
                "Sorgu embedding boyutu "
                f"{self.dimension} olmalı, alınan {query_embedding.shape}"
            )

        result_count = min(max(1, int(top_k)), self.index.ntotal)
        scores, indices = self.index.search(query_embedding, result_count)
        results = []
        for score, index in zip(scores[0], indices[0]):
            if index < 0 or index >= len(self.metadata):
                continue
            score = float(score)
            if score < threshold:
                continue
            metadata = self.metadata[index]
            results.append({
                "text": metadata["text"],
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk_index": metadata["chunk_index"],
                "score": score,
            })
        results.sort(key=lambda result: result["score"], reverse=True)
        return results

    def save(self):
        os.makedirs(VECTORSTORE_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_FILE)
        with open(METADATA_FILE, "w", encoding="utf-8") as file:
            json.dump(self.metadata, file, ensure_ascii=False, indent=2)

    def load(self):
        index = faiss.read_index(INDEX_FILE)
        if index.d != self.dimension:
            raise ValueError(
                f"İndeks {index.d} boyutlu, mevcut model {self.dimension} boyutlu. "
                "Model değiştiğinde eski indeks kullanılamaz. Silin: "
                f"{INDEX_FILE} ve {METADATA_FILE}, sonra '--ingest' çalıştırın."
            )
        with open(METADATA_FILE, "r", encoding="utf-8") as file:
            metadata = json.load(file)
        if index.ntotal != len(metadata):
            raise ValueError("FAISS indeksi ve metadata sayıları eşleşmiyor")
        self.index = index
        self.metadata = metadata

    def get_total(self):
        return self.index.ntotal

    def is_empty(self):
        return self.index.ntotal == 0
