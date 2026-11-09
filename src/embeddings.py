from functools import cache

import numpy as np
from sentence_transformers import SentenceTransformer

DEFAULT_BI_ENCODER = "sentence-transformers/all-MiniLM-L6-v2"


@cache
def load_bi_encoder(model_name: str = DEFAULT_BI_ENCODER) -> SentenceTransformer:
    try:
        return SentenceTransformer(model_name)
    except RuntimeError:
        return SentenceTransformer(model_name, device="cpu")


def embed_texts(
    texts: list[str],
    model_name: str = DEFAULT_BI_ENCODER,
    batch_size: int = 32,
) -> np.ndarray:
    model = load_bi_encoder(model_name)
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(embeddings, dtype=np.float32)