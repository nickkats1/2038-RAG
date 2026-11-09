from src.retrievers.bm25 import Bm25Retriever
from src.retrievers.dense import DenseRetriever
from src.data_utils import document_key


class HybridRetriever:
    def __init__(
        self,
        bm25=None,
        dense=None,
        bm25_weight=0.1,
        dense_weight=0.6,
        candidates=40,
        rrf_k=60,
    ):
        self.bm25 = bm25 if bm25 is not None else Bm25Retriever()
        self.dense = dense if dense is not None else DenseRetriever()
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.candidates = candidates
        self.rrf_k = rrf_k

    def add_documents(self, documents):
        self.bm25.add_documents(documents)
        self.dense.add_documents(documents)

    def retrieve(self, query, top_k):
        if top_k < 1:
            return []

        rankings = [
            (self.bm25.retrieve(query, top_k=self.candidates), self.bm25_weight),
            (self.dense.retrieve(query, top_k=self.candidates), self.dense_weight),
        ]

        scores = {}
        documents = {}

        for ranking, weight in rankings:
            if weight <= 0:
                continue

            for rank, doc in enumerate(ranking, start=1):
                key = document_key(doc)
                documents[key] = doc
                scores[key] = (
                    scores.get(key, 0.0)
                    + weight / (self.rrf_k + rank)
                )

        ranked_ids = sorted(scores, key=scores.get, reverse=True)
        return [documents[key] for key in ranked_ids[:top_k]]