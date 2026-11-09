from collections.abc import Callable

import numpy as np

from src.embeddings import DEFAULT_BI_ENCODER, embed_texts
from src.graph.extraction import normalise


class EntityResolver:
    def __init__(
        self,
        threshold: float = 0.75,
        model_name: str = DEFAULT_BI_ENCODER,
        embed: Callable[[list[str], str], np.ndarray] = embed_texts,
    ) -> None:
        """Start with no known entities.

        Args:
            threshold: cosine similarity at or above which a mention joins an
                existing key.
            model_name: bi-encoder used to compare mentions.
            embed: encoder to compare mentions with.
        """
        self.threshold = threshold
        self.model_name = model_name
        self.embed = embed
        self._keys: list[str] = []
        self._vectors: np.ndarray | None = None
        self._canonical: dict[str, str] = {}
        self._pending: dict[str, np.ndarray] = {}

    def prepare(self, mentions: list[str]) -> None:
        keys = {
            key
            for key in map(normalise, mentions)
            if key and key not in self._canonical and key not in self._pending
        }
        if not keys:
            return
        ordered = sorted(keys)
        for key, vector in zip(ordered, self.embed(ordered, self.model_name)):
            self._pending[key] = vector[np.newaxis, :]

    def resolve(self, mention: str) -> str:
        key = normalise(mention)
        if not key or key in self._canonical:
            return self._canonical.get(key, key)

        vector = self._pending.pop(key, None)
        if vector is None:
            vector = self.embed([key], self.model_name)
        if self._vectors is not None:
            similarities = self._vectors @ vector[0]
            closest = int(similarities.argmax())
            if similarities[closest] >= self.threshold:
                self._canonical[key] = self._keys[closest]
                return self._canonical[key]

        self._keys.append(key)
        self._vectors = (
            vector if self._vectors is None else np.vstack([self._vectors, vector])
        )
        self._canonical[key] = key
        return key
