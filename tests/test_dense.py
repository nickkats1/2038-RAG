import numpy as np
from langchain_core.documents import Document

from src.retrievers.dense import DenseRetriever


def test_dense_retrieval():
    vectors = {
        "apple orchard": [1.0, 0.0],
        "diesel engine": [0.0, 1.0],
        "fruit": [0.8, 0.6],
    }

    def embed(texts, model_name):
        return np.array([vectors[text] for text in texts], dtype=np.float32)

    retriever = DenseRetriever(embed=embed)
    documents = [
        Document(page_content="diesel engine", metadata={"page": 1}),
        Document(page_content="apple orchard", metadata={"page": 2}),
    ]

    assert retriever.retrieve("fruit", top_k=1) == []
    retriever.add_documents(documents)
    assert retriever.retrieve("fruit", top_k=1) == [documents[1]]
    assert retriever.retrieve("fruit", top_k=10) == documents[::-1]
