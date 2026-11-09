import pytest
from langchain_core.documents import Document

from src.retrievers.bm25 import Bm25Retriever


class TestBM25Retriever:
    def test_adds_and_retrieves_documents(self):
        documents = [
            Document(page_content="apple orchard fruit", metadata={"page": 0}),
            Document(page_content="diesel engine truck", metadata={"page": 1}),
        ]
        retriever = Bm25Retriever()
        retriever.add_documents(documents)
        assert retriever.documents == documents
        assert retriever.retrieve("APPLE!", top_k=10) == [documents[0]]
        assert retriever.retrieve("engine", top_k=1) == [documents[1]]
        assert retriever.retrieve("unknownword", top_k=10) == []

    def test_ranks_and_limits_results(self):
        documents = [
            Document(page_content="apple banana banana"),
            Document(page_content="apple apple apple"),
            Document(page_content="diesel engine truck"),
        ]
        retriever = Bm25Retriever()
        retriever.add_documents(documents)
        assert retriever.retrieve("apple", top_k=10) == [documents[1], documents[0]]
        assert retriever.retrieve("apple", top_k=1) == [documents[1]]

    def test_adding_documents_preserves_previous_results(self):
        first = Document(page_content="apple orchard")
        second = Document(page_content="diesel engine")
        retriever = Bm25Retriever()
        retriever.add_documents([first])
        retriever.add_documents([])
        retriever.add_documents([second])
        assert retriever.retrieve("apple", top_k=10) == [first]
        assert retriever.retrieve("diesel", top_k=10) == [second]

    def test_empty_index_and_blank_documents(self):
        retriever = Bm25Retriever()
        assert retriever.retrieve("apple", top_k=10) == []
        retriever.add_documents([])
        retriever.add_documents([Document(page_content=""), Document(page_content="!!!")])
        assert retriever.retrieve("apple", top_k=10) == []
        document = Document(page_content="apple orchard")
        retriever.add_documents([document])
        assert retriever.retrieve("apple", top_k=10) == [document]

    @pytest.mark.parametrize("query,top_k", [("", 1), ("!!!", 1), ("apple", 0), ("apple", -1)])
    def test_empty_queries_and_nonpositive_limits(self, query, top_k):
        retriever = Bm25Retriever()
        retriever.add_documents([Document(page_content="apple orchard")])
        assert retriever.retrieve(query, top_k=top_k) == []
