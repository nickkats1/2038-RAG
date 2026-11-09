from collections import Counter

import networkx as nx
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from src.retrievers.bm25 import tokenize
from src.graph.extraction import Triple, extract_triples
from src.graph.resolution import EntityResolver


def _contains_run(terms: list[str], run: list[str]) -> bool:
    if not run:
        return False
    return any(
        terms[start : start + len(run)] == run
        for start in range(len(terms) - len(run) + 1)
    )


class GraphRetriever:
    def __init__(
        self,
        llm: ChatOpenAI,
        hops: int = 3,
        resolver: EntityResolver | None = None,
    ) -> None:
        if hops < 0:
            raise ValueError("hops must be nonnegative")
        self.llm = llm
        self.hops = hops
        self.resolver = resolver if resolver is not None else EntityResolver()
        self.graph = nx.Graph()
        self.documents: list[Document] = []
        self._mentions: dict[str, set[int]] = {}

    def add_documents(self, documents: list[Document]) -> None:
        if not documents:
            return
        extracted = [extract_triples(doc.page_content, self.llm) for doc in documents]
        self.resolver.prepare(
            [name for triples in extracted for t in triples for name in (t.subject, t.object)]
        )
        start = len(self.documents)
        for offset, triples in enumerate(extracted):
            for triple in triples:
                self._add_triple(triple, start + offset)
        self.documents.extend(documents)

    def _add_triple(self, triple: Triple, doc_index: int) -> None:
        subject = self.resolver.resolve(triple.subject)
        obj = self.resolver.resolve(triple.object)
        self.graph.add_edge(subject, obj, relation=triple.relation)
        for entity in (subject, obj):
            self._mentions.setdefault(entity, set()).add(doc_index)

    def _entities_named_in(self, query: str) -> set[str]:
        terms = tokenize(query)
        return {
            entity
            for entity in self.graph.nodes
            if _contains_run(terms, tokenize(entity))
        }

    def _rank(self, entities: set[str]) -> list[int]:
        counts: Counter[int] = Counter()
        for entity in entities:
            counts.update(self._mentions.get(entity, set()))
        return sorted(counts, key=lambda index: (-counts[index], index))

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        if top_k <= 0:
            return []
        reached: set[str] = set()
        for entity in self._entities_named_in(query):
            reached |= set(
                nx.single_source_shortest_path_length(
                    self.graph, entity, cutoff=self.hops
                )
            )
        return [self.documents[index] for index in self._rank(reached)[:top_k]]

