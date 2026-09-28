from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


class Embedder:
    def __init__(self, model_name=EMBEDDING_MODEL):
        self.model = SentenceTransformer(model_name)

    def encode(
        self,
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize=True,
    ):
        texts = [f"passage: {text}" for text in texts]
        return self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize,
            show_progress_bar=True,
        )

    def encode_query(self, text):
        return self.model.encode(
            [f"query: {text}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

    def encode_single(self, text):
        return self.encode_query(text)
