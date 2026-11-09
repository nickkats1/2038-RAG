from unittest.mock import Mock

import numpy as np
import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from src.graph.extraction import Triple, extract_triples, parse_triples
from src.graph.resolution import EntityResolver
from src.graph.retriever import GraphRetriever
from src.prompts import TRIPLE_PROMPT


def make_retriever(hops=1):
    names = ["ada", "ben", "cy"]
    def embed(texts, model_name):
        return np.array([np.eye(len(names))[names.index(t)] for t in texts])
    llm = Mock()
    resolver = EntityResolver(embed=embed)
    return GraphRetriever(llm=llm, hops=hops, resolver=resolver), llm


def test_extracts_chat_response_text():
    llm = Mock()
    llm.invoke.return_value = AIMessage(content=
        '```json\n[{"subject":"Ada","relation":"knows","object":"Ben"}]\n```')
    assert extract_triples("Ada knows Ben.", llm) == [Triple("Ada", "knows", "Ben")]
    llm.invoke.assert_called_once_with(TRIPLE_PROMPT.format(text="Ada knows Ben."))


def test_indexes_chunks_and_traverses_relationships():
    retriever, llm = make_retriever()
    docs = [Document(page_content="Ada knows Ben.", metadata={"page": 1}),
            Document(page_content="Ben knows Cy.", metadata={"page": 2})]
    llm.invoke.side_effect = [
        AIMessage(content='[{"subject":"Ada","relation":"knows","object":"Ben"}]'),
        AIMessage(content='[{"subject":"Ben","relation":"knows","object":"Cy"}]')]
    retriever.add_documents(docs[:1])
    retriever.add_documents([])
    retriever.add_documents(docs[1:])
    assert llm.invoke.call_count == 2
    assert llm.invoke.call_args_list[0].args[0] == TRIPLE_PROMPT.format(text=docs[0].page_content)
    assert llm.invoke.call_args_list[1].args[0] == TRIPLE_PROMPT.format(text=docs[1].page_content)
    assert retriever.retrieve("ADA!", top_k=5) == docs
    assert retriever.retrieve("Ada", top_k=1) == docs[:1]
    assert retriever.retrieve("Cy", top_k=5) == docs[::-1]
    assert retriever.retrieve("Nobody", top_k=5) == []
    assert retriever.retrieve("", top_k=5) == []
    assert retriever.retrieve("Ada", top_k=0) == []
    assert retriever.retrieve("Ada", top_k=-1) == []
    retriever.hops = 0
    assert retriever.retrieve("Ada", top_k=5) == docs[:1]


def test_empty_extraction_keeps_document_indices_aligned():
    retriever, llm = make_retriever()
    docs = [Document(page_content="No relationships."), Document(page_content="Ada knows Ben.")]
    llm.invoke.side_effect = [AIMessage(content="[]"), AIMessage(content=
        '[{"subject":"Ada","relation":"knows","object":"Ben"}]')]
    assert retriever.retrieve("Ada") == []
    retriever.add_documents(docs)
    assert retriever.retrieve("Ada") == docs[1:]


@pytest.mark.parametrize("reply", ["invalid JSON", "{}", '[{"subject":"Ada"}]'])
def test_rejects_invalid_triples(reply):
    assert parse_triples(reply) == []


def test_negative_hops_rejected():
    with pytest.raises(ValueError, match="hops"):
        make_retriever(hops=-1)
