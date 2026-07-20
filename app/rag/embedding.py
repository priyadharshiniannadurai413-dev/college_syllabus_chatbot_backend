from sentence_transformers import SentenceTransformer

# Load model once at module level — avoids re-loading weights on every request
_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


class EmbeddingModel:
    def __init__(self):
        self.model = _model

    def generate_embeddings(self, chunks):
        embeddings = self.model.encode(
            chunks,
            convert_to_numpy=True
        )

        return embeddings.tolist()
