"""HyDE without embeddings: a hypothetical answer, expanded by Rocchio, not concatenated.

The problem this addresses. On 2,996 Chinese statutes, a query copied from a clause
retrieves its statute at hit@1 0.87; a real user question retrieves it at 0.12. The user
says "公司没给我做安全培训就让我直接上岗，这样允许吗？" while the statute says
"生产经营单位应当对从业人员进行安全生产教育和培训". A lexical index has almost no overlap to
work with.

The idea. Ask the model to write the passage that WOULD answer the question - in the
register of the corpus - and search with that too. HyDE, without needing an embedding to
compare vectors: the hypothetical document is turned into term weights.

The measured trap. Appending the generated text to the query string is the obvious
implementation and it barely works: SIGIR'26 (Jedidi & Lin) measured Recall@20 across 14
datasets at 28.8 for BM25, 29.5 for naive concatenation, and 34.7 for Rocchio feedback
over the same generated text. The generated passage contains useful terms and noise, and
concatenation gives the noise full weight. Rocchio weights the terms instead.

Rocchio, for query q, feedback set F (the hypothetical passage's terms) and beta:
    w(t) = tftd(t, q) + beta * tftd(t, F)
with terms that occur in more than MAX_DF_RATIO of the corpus dropped, because a term
everywhere discriminates nowhere. This is a closed-form arithmetic step - no second
model, no training, consistent with the project's single-model constraint.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.exceptions import ModelOutputError
from app.model_client.generation import GenerationClient

logger = logging.getLogger(__name__)

# Feedback strength. The SIGIR'26 result used beta around 0.75 with the original query
# retaining its weight; alpha = 1.0 by construction since the query is never downweighted.
BETA = 0.75
# Rocchio's feedback depth: the paper found 32-64 terms sufficient, saturating after that.
MAX_FEEDBACK_TERMS = 48
# A term appearing in more than this fraction of documents is corpus stopword-ish and
# adds noise rather than discrimination.
MAX_DF_RATIO = 0.10
MAX_HYPOTHETICAL_CHARS = 600

PROMPT = """Write the passage that would answer the question below.

Rules:
- Write it in the register of an official document (statute, regulation, contract or
  manual) - formal, declarative, the way the source material is actually written.
- Do NOT answer the question directly and do NOT hedge or comment. Write the passage
  itself, as if it were an excerpt from such a document.
- Use the terminology that such a document would use for this subject matter, including
  formal synonyms for the everyday words in the question.
- At most 200 words. If you do not know the specifics, write the kind of rule that
  would apply, without inventing names, numbers or dates.

Question: __QUERY__

Passage:"""


def _render(query: str) -> str:
    return PROMPT.replace("__QUERY__", query)


async def hypothetical_document(
    query: str, generation_client: GenerationClient
) -> str:
    """The passage the model believes would answer ``query``, in corpus register."""
    result = await generation_client.generate(
        [{"role": "user", "content": _render(query)}], temperature=0
    )
    content = str(result.get("content") or "").strip()
    if not content:
        raise ModelOutputError(
            "hypothetical document generation returned no content; a reasoning model "
            "may have spent its budget before writing - check the token cap"
        )
    return content[:MAX_HYPOTHETICAL_CHARS]


def rocchio_weights(
    query_terms: list[str],
    feedback_terms: list[str],
    *,
    beta: float = BETA,
    max_feedback_terms: int = MAX_FEEDBACK_TERMS,
    document_frequencies: dict[str, int] | None = None,
    n_docs: int = 0,
    max_df_ratio: float = MAX_DF_RATIO,
) -> dict[str, float]:
    """Rocchio term weights from the query and a feedback term bag.

    ``document_frequencies``/``n_docs`` enable the corpus-frequency cutoff. Passing them
    is optional so this stays unit-testable, but the caller should pass them: without
    the cutoff, a generated passage contributes terms that appear in most documents,
    which is where naive concatenation loses its signal.
    """
    weights: dict[str, float] = {}
    for term in query_terms:
        weights[term] = weights.get(term, 0.0) + 1.0

    counts: dict[str, int] = {}
    for term in feedback_terms:
        counts[term] = counts.get(term, 0) + 1

    if document_frequencies is not None and n_docs > 0:
        cutoff = max_df_ratio * n_docs
        dropped = 0
        for term in list(counts):
            df = document_frequencies.get(term)
            if df is not None and df > cutoff:
                del counts[term]
                dropped += 1
        if dropped:
            logger.info(
                "rocchio: dropped %d feedback term(s) above the %.0f%% corpus-frequency "
                "cutoff",
                dropped,
                max_df_ratio * 100,
            )

    # Keep the strongest feedback terms only: the paper's saturation point.
    ranked = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    for term, count in ranked[:max_feedback_terms]:
        weights[term] = weights.get(term, 0.0) + beta * count
    return weights


async def build_hyde_weights(
    query: str,
    generation_client: GenerationClient,
    *,
    tokenize: Any,
    document_frequencies: dict[str, int] | None = None,
    n_docs: int = 0,
) -> dict[str, float]:
    """Generate the hypothetical passage and turn it into weighted query terms."""
    hypothetical = await hypothetical_document(query, generation_client)
    query_terms = list(tokenize(query))
    feedback_terms = list(tokenize(hypothetical))
    weights = rocchio_weights(
        query_terms,
        feedback_terms,
        document_frequencies=document_frequencies,
        n_docs=n_docs,
    )
    logger.info(
        "hyde: %d query term(s) + %d feedback term(s) -> %d weighted term(s)",
        len(query_terms),
        len(feedback_terms),
        len(weights),
    )
    return weights
