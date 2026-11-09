import json
import re
from dataclasses import dataclass
from langchain_openai import ChatOpenAI

from src.prompts import TRIPLE_PROMPT

_WHITESPACE = re.compile(r"\s+")
_REQUIRED_KEYS = ("subject", "relation", "object")


def normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip().lower()


@dataclass(frozen=True)
class Triple:
    subject: str
    relation: str
    object: str


def _strip_code_fence(reply: str) -> str:
    text = reply.strip()
    if not text.startswith("```"):
        return text
    without_opening = text.split("\n", 1)[-1]
    closing = without_opening.rfind("```")
    return without_opening[:closing] if closing != -1 else without_opening


def _to_triple(entry: object) -> Triple | None:
    if not isinstance(entry, dict) or not all(
        isinstance(entry.get(key), str) for key in _REQUIRED_KEYS
    ):
        return None
    return Triple(entry["subject"], entry["relation"], entry["object"])


def parse_triples(reply: str) -> list[Triple]:
    try:
        entries = json.loads(_strip_code_fence(reply))
    except json.JSONDecodeError:
        return []
    if not isinstance(entries, list):
        return []
    return [triple for triple in map(_to_triple, entries) if triple is not None]


def extract_triples(text: str, llm: ChatOpenAI) -> list[Triple]:
    reply = llm.invoke(TRIPLE_PROMPT.format(text=text))
    return parse_triples(reply.text)
