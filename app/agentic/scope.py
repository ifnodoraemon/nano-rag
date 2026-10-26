"""Scope is a document property, read from the text - not a per-domain table.

The case that forced this. A user asks "男的多少岁能结婚？" - the general rule. The
retriever returned three autonomous-region derogations instead, because their articles
say 结婚年龄 verbatim while the national statute phrases it as 结婚年龄，男不得早于二十二周岁。
The model refused correctly, but retrieval had already lost: it treated a regional
variation and a national law as equally good answers to a question about the general rule.

The trap in fixing that. The obvious fix is a table of "national" issuing bodies
(全国人民代表大会, 国务院 ...). That is a special case dressed as a rule: it does nothing for
a contract, a manual, an English policy, or any corpus whose issuing bodies are named
differently. This project forbids hardcoded business meaning, and a ranking of which
institutions outrank which is exactly that.

What is actually generic: every document states, somewhere, where it applies and how
authoritative it is. A statute names its legislature; a contract names the group entities
and territories it binds; a manual names the product, site or business unit; a revision
says it supersedes an earlier edition. Those statements are scope, in the document's own
words, whatever the domain.

So the model reads each document once at ingest and records:

  applies_to      free-text statements of where the document applies ("本省", "中国大陆
                  及香港地区子公司", "适用于所有生产基地")
  authority_rank  where it sits in its own hierarchy as an integer, 1 = the most general
                  instrument in that corpus

and at query time the model says whether the question is about the general case or a
specific one. Ranking then prefers matching documents by a MULTIPLIER - never a filter,
because a regional rule is a wrong answer to a general question but the right answer to a
regional one, and a filter would make it unfindable.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence

from app.core.exceptions import ModelOutputError
from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)

# How much a scope-matching document is boosted, and a mismatching one damped. Both stay
# above zero: these reorder, they never remove.
MATCH_FACTOR = 1.25
MISMATCH_FACTOR = 0.90

DOCUMENT_SCOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "applies_to": {
            "type": "array",
            "items": {"type": "string"},
            "description": (
                "where the document says it applies, in its own words - territories, "
                "entities, sites, products, audiences; empty if it states no scope"
            ),
        },
        "authority_rank": {
            "type": "integer",
            "description": (
                "1 = the most general / highest instrument for this subject in this "
                "corpus; larger = more specific or lower. Infer from the document's own "
                "signals (who issued it, what it derives from, what it supersedes). "
                "Omit when the document gives no signal."
            ),
        },
    },
    "required": ["applies_to"],
}

DOCUMENT_SCOPE_PROMPT = """Read this document and record its SCOPE - where it applies and
how authoritative it is. Do not summarise the document.

applies_to: list where the document itself says it applies. Quote or closely paraphrase
its own wording - territories, regions, entities, business units, products, audiences.
If it applies everywhere or states no limit, return [].

authority_rank: 1 if this looks like the most general / highest-level instrument for its
subject (a national law, a group-wide policy, a master contract); 2, 3 ... for
increasingly specific ones (regional rules, subsidiary policies, appendices). Judge from
the document's own signals: who issued it, what it derives from, which it supersedes. If
the document gives no signal, omit the field.

Title: __TITLE__

DOCUMENT (excerpt):
__TEXT__

Return only JSON: {"applies_to": ["..."], "authority_rank": 1}"""

QUERY_SCOPE_SCHEMA = {
    "type": "object",
    "properties": {
        "scope": {
            "type": "string",
            "enum": ["general", "specific", "unspecified"],
            "description": (
                "general: asks for the rule that applies broadly, naming no particular "
                "place/entity; specific: names a place, region, entity or product, or "
                "asks about local/specific rules; unspecified: neither"
            ),
        }
    },
    "required": ["scope"],
}

QUERY_SCOPE_PROMPT = """Does this question ask for the general rule, or about a specific
place/entity/product?

- general: "男的多少岁能结婚", "劳动合同最长试用期多久" - asks for the broadly applicable rule
- specific: "宁夏的生育假多少天", "华东区的报销标准" - names a scope
- unspecified: it does not say

Do not answer the question. Do not invent a scope that is not mentioned.

Question: __QUERY__

Return only JSON: {"scope": "general" | "specific" | "unspecified"}"""

MAX_EXCERPT_CHARS = 6_000


@dataclass(frozen=True)
class ScopePreference:
    """Multipliers for a document whose scope matches, or does not match, the question."""

    matches: float = 1.0
    others: float = 1.0

    @property
    def is_neutral(self) -> bool:
        return self.matches == 1.0 and self.others == 1.0


NEUTRAL = ScopePreference()


def _render_document_prompt(title: str, excerpt: str) -> str:
    # str.replace, not .format: the prompt carries a literal JSON example.
    return DOCUMENT_SCOPE_PROMPT.replace("__TITLE__", title).replace("__TEXT__", excerpt)


async def describe_document_scope(
    text: str,
    generation_client: GenerationClient,
    *,
    title: str = "",
) -> dict[str, Any]:
    """The document's own scope statement, read by the model at ingest time.

    Returns ``{"applies_to": [...], "authority_rank": int | None}``. An empty
    ``applies_to`` with no rank means the document states no scope - a normal answer, not
    a failure, so it does not raise. A malformed reply does raise.
    """
    from app.utils.json_utils import parse_json_object

    excerpt = text.strip()[:MAX_EXCERPT_CHARS]
    if not excerpt:
        return {"applies_to": [], "authority_rank": None}

    result = await generation_client.generate(
        [{"role": "user", "content": _render_document_prompt(title, excerpt)}],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "document_scope", "schema": DOCUMENT_SCOPE_SCHEMA},
        },
    )
    try:
        payload = parse_json_object(str(result.get("content") or ""))
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise ModelOutputError(
            f"document scope extraction returned unparseable output for {title!r}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelOutputError(
            f"document scope extraction returned a non-object payload for {title!r}"
        )
    raw_scope = payload.get("applies_to")
    if raw_scope is None:
        raw_scope = []
    if not isinstance(raw_scope, list):
        raise ModelOutputError(
            f"document scope extraction returned non-list applies_to for {title!r}"
        )
    applies_to = [
        " ".join(item.split())
        for item in raw_scope
        if isinstance(item, str) and item.strip()
    ]
    rank = payload.get("authority_rank")
    if rank is not None and (
        isinstance(rank, bool) or not isinstance(rank, int) or rank < 1
    ):
        raise ModelOutputError(
            f"document scope extraction returned an invalid authority_rank for "
            f"{title!r}: {rank!r}"
        )
    return {"applies_to": applies_to, "authority_rank": rank}


async def query_scope(query: str, generation_client: GenerationClient) -> str:
    """Whether the question is about the general case, a specific one, or neither."""
    from app.utils.json_utils import parse_json_object

    result = await generation_client.generate(
        [{"role": "user", "content": QUERY_SCOPE_PROMPT.replace("__QUERY__", query)}],
        temperature=0,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "query_scope", "schema": QUERY_SCOPE_SCHEMA},
        },
    )
    try:
        payload = parse_json_object(str(result.get("content") or ""))
    except Exception as exc:  # noqa: BLE001 - re-raised with context
        raise ModelOutputError(
            f"query scope classification returned unparseable output: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ModelOutputError("query scope classification returned a non-object payload")
    verdict = payload.get("scope")
    if verdict not in {"general", "specific", "unspecified"}:
        raise ModelOutputError(
            f"query scope classification returned an unknown verdict: {verdict!r}"
        )
    logger.info("query scope for %r: %s", query, verdict)
    return str(verdict)


def preference_for(verdict: str) -> ScopePreference:
    """Ranking preference for a query-scope verdict; neutral when unspecified."""
    if verdict in {"general", "specific"}:
        return ScopePreference(matches=MATCH_FACTOR, others=MISMATCH_FACTOR)
    return NEUTRAL


def document_matches(metadata: dict[str, Any], verdict: str) -> bool:
    """Whether a document's recorded scope matches the query's scope.

    Generic rule, no domain vocabulary:
    - a general question matches documents that state NO narrow scope and rank highest;
    - a specific question matches documents that DO state a scope.
    A document carrying neither field is treated as matching, because nothing is known
    about it and guessing would be worse than neutrality.
    """
    applies_to: Sequence[Any] = metadata.get("applies_to") or []
    raw_rank = metadata.get("authority_rank")
    rank = raw_rank if isinstance(raw_rank, int) and not isinstance(raw_rank, bool) else None
    if not applies_to and rank is None:
        return True
    if verdict == "general":
        return not applies_to and (rank is None or rank <= 1)
    if verdict == "specific":
        return bool(applies_to) or (rank is not None and rank > 1)
    return True
