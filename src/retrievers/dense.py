from collections.abc import Callable

import faiss
import numpy as np
from langchain_core.documents import Document

from src.embeddings import DEFAULT_BI_ENCODER, embed_texts

class DenseRetriever:

    def __init__(
        self,
        model_name: str = DEFAULT_BI_ENCODER,
        embed: Callable[[list[str], str], np.ndarray] = embed_texts,
    ) -> None:
        self.model_name = model_name
        self.embed = embed
        self.documents: list[Document] = []
        self.index: faiss.Index | None = None


    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        vectors = self.embed([doc.page_content for doc in documents], self.model_name)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.documents.extend(documents)

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        if self.index is None:
            return []
        query_vector = self.embed([query], self.model_name)
        _, positions = self.index.search(query_vector, min(top_k, len(self.documents)))
        return [self.documents[i] for i in positions[0] if i >= 0]