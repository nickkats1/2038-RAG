import re

import bm25s
from langchain_core.documents import Document
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_WORD = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [word for word in _WORD.findall(text.lower()) if word not in ENGLISH_STOP_WORDS]


class Bm25Retriever:
    def __init__(self) -> None:
        self.documents: list[Document] = []
        self._bm25_retriever: bm25s.BM25 | None = None

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        self.documents.extend(documents)
        corpus = [tokenize(doc.page_content) for doc in self.documents]
        if any(corpus):
            self._bm25_retriever = bm25s.BM25()
            self._bm25_retriever.index(corpus, show_progress=False)

    def retrieve(self, query: str, top_k: int) -> list[Document]:
        terms = tokenize(query)
        if self._bm25_retriever is None or not terms or top_k <= 0:
            return []
        scores = self._bm25_retriever.get_scores(terms)
        matched = [i for i, score in enumerate(scores) if score > 0]
        matched.sort(key=lambda i: -scores[i])
        return [self.documents[i] for i in matched[:top_k]]
