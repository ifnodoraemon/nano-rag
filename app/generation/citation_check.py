"""Deterministic citation integrity: is every claim actually attributable?

"Every answer is traceable to retrieved context" was enforced only by the prompt
asking the model to cite ``[C1]`` - nothing checked that the label exists or that
the cited passage contains the claim. A model can cite a label that was never
provided, or attach a real label to a number that appears nowhere in it, and the
answer still looks traceable.

This module checks, with no model call:

1. every cited label resolves to a context that was actually provided
   (a label that does not exist is a fabricated citation);
2. the numbers stated in a cited sentence appear in the context(s) that sentence
   cites - numbers are the least ambiguous claim fragment and the most damaging
   to get wrong (prices, deadlines, versions);
3. optionally, quoted spans appear verbatim in the cited text.

Multi-sentence and multi-citation cases are handled per sentence: a sentence may
cite several labels, and a number counts as grounded if it appears in any of
them (the model is asked to cite the label(s) supporting the sentence, not to
map each number to exactly one).

The score is reported as ``None`` when the answer carries no citations at all -
that is "not evaluable", not "perfect". Callers see ``sentences_without_citations``
so a silently uncited answer is visible instead of scoring 1.0.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Labels as the formatter emits them: [C1] or a grouped [C1, C2].
CITATION_GROUP_RE = re.compile(r"\[((?:C\d+\s*,\s*)*C\d+)\]")
CITATION_LABEL_RE = re.compile(r"C\d+")
# Sentence boundaries differ by script: Chinese terminators (。！？；) do not
# require a following space, while ASCII ones do (splitting "15.5" on the dot
# would invent a sentence boundary). The formatter's shared regex requires
# whitespace after ANY terminator, which merged an entire Chinese paragraph into
# one "sentence" and made sentence-level citation scoring meaningless.
SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？；])|(?<=[.!?;])(?=\s|$)|(?<=\n)")
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
QUOTED_SPAN_RE = re.compile(r"[\"“”「」『』]([^\"“”「」『』]{2,80})[\"“”「」『』]")


@dataclass(frozen=True)
class CitationIssue:
    kind: str  # unresolved_label | ungrounded_number | ungrounded_quote
    detail: str
    sentence: str
    label: str | None = None


@dataclass(frozen=True)
class CitationIntegrityReport:
    cited_labels: tuple[str, ...] = ()
    resolved_labels: tuple[str, ...] = ()
    unresolved_labels: tuple[str, ...] = ()
    sentences_with_citations: int = 0
    sentences_without_citations: int = 0
    grounded_sentences: int = 0
    ungrounded_number_count: int = 0
    ungrounded_quote_count: int = 0
    # Relevance of the cited evidence to the sentence, independent of label
    # resolution. A sentence can cite a label that resolves and whose numbers
    # appear - while the context is about a different subject. Integrity alone
    # cannot see that, so precision is measured separately.
    sentences_lexically_supported: int = 0
    issues: tuple[CitationIssue, ...] = ()

    @property
    def citation_precision(self) -> float | None:
        """Fraction of cited sentences whose cited evidence is lexically related.

        A deterministic proxy: it cannot judge paraphrase, but it does catch a
        citation pointing at an unrelated passage, which label-and-number
        grounding alone reports as perfectly fine.
        """
        if self.sentences_with_citations == 0:
            return None
        return round(
            self.sentences_lexically_supported / self.sentences_with_citations, 4
        )

    @property
    def integrity_score(self) -> float | None:
        """Fraction of cited sentences that are fully supported.

        ``None`` when the answer cites nothing: that is a different problem
        (visible via ``sentences_without_citations``), not a perfect score.
        """
        if self.sentences_with_citations == 0:
            return None
        return round(
            self.grounded_sentences / self.sentences_with_citations, 4
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "integrity_score": self.integrity_score,
            "cited_labels": list(self.cited_labels),
            "resolved_labels": list(self.resolved_labels),
            "unresolved_labels": list(self.unresolved_labels),
            "sentences_with_citations": self.sentences_with_citations,
            "sentences_without_citations": self.sentences_without_citations,
            "grounded_sentences": self.grounded_sentences,
            "ungrounded_number_count": self.ungrounded_number_count,
            "ungrounded_quote_count": self.ungrounded_quote_count,
            "citation_precision": self.citation_precision,
            "sentences_lexically_supported": self.sentences_lexically_supported,
            "issues": [
                {
                    "kind": issue.kind,
                    "label": issue.label,
                    "detail": issue.detail,
                    "sentence": issue.sentence,
                }
                for issue in self.issues
            ],
        }


MIN_CITATION_SUPPORT = 0.34


def _content_tokens(text: str) -> set[str]:
    """Tokens that carry meaning: CJK characters and alphanumeric runs.

    Uses the same CJK tokenizer as the discovery index so the check does not
    grow a second, private notion of what a term is.
    """
    from app.retrieval.tokenizer import tokenize

    return {token for token in tokenize(text.lower()) if token.strip()}


def _lexically_supported(sentence: str, evidence: str) -> bool:
    """Does the cited evidence share enough substance with the claim?

    Deliberately lenient (a third of the sentence's tokens): the point is to catch
    a citation pointing at an unrelated passage, not to grade paraphrase quality -
    that is what the LLM-assisted metrics are for.
    """
    claim_tokens = _content_tokens(CITATION_GROUP_RE.sub(" ", sentence))
    if not claim_tokens:
        return True
    evidence_tokens = _content_tokens(evidence)
    if not evidence_tokens:
        return False
    overlap = len(claim_tokens & evidence_tokens) / len(claim_tokens)
    return overlap >= MIN_CITATION_SUPPORT


def _context_texts_by_label(
    contexts: list[dict[str, object]],
) -> dict[str, str]:
    """Map citation label -> the text it points at.

    A label that appears on several contexts keeps all their text: the model
    cited the label, so evidence in any of them supports the sentence.
    """
    texts: dict[str, str] = {}
    for context in contexts:
        label = context.get("citation_label")
        if label is None:
            continue
        key = str(label).strip().upper()
        text = str(context.get("text") or context.get("span_text") or "")
        if key in texts:
            texts[key] = f"{texts[key]}\n{text}"
        else:
            texts[key] = text
    return texts


def _normalize_number(raw: str) -> str:
    """Compare numbers by value, so 15, 15.0 and 15.00 are the same claim."""
    cleaned = raw.replace(",", " ")
    cleaned = re.sub(r"\s+", "", cleaned)
    try:
        value = float(cleaned)
    except ValueError:
        return cleaned
    if value == int(value):
        return str(int(value))
    return repr(value)


def _numbers_in(text: str) -> set[str]:
    return {
        _normalize_number(match)
        for match in NUMBER_RE.findall(CITATION_GROUP_RE.sub(" ", text))
    }


def _cited_labels_in(sentence: str) -> list[str]:
    labels: list[str] = []
    for group in CITATION_GROUP_RE.findall(sentence):
        for label in CITATION_LABEL_RE.findall(group):
            key = label.strip().upper()
            if key not in labels:
                labels.append(key)
    return labels


def check_citations(
    answer: str,
    contexts: list[dict[str, object]],
    *,
    check_quotes: bool = True,
) -> CitationIntegrityReport:
    """Verify the answer's citation labels and the numbers they claim to support."""
    text_by_label = _context_texts_by_label(contexts)
    provided = set(text_by_label)

    cited: list[str] = []
    resolved: list[str] = []
    unresolved: list[str] = []
    issues: list[CitationIssue] = []
    sentences_with = 0
    sentences_without = 0
    grounded = 0
    ungrounded_numbers = 0
    ungrounded_quotes = 0
    lexically_supported = 0

    for sentence in SENTENCE_SPLIT_RE.split(str(answer or "")):
        sentence = sentence.strip()
        if not sentence:
            continue
        labels = _cited_labels_in(sentence)
        if not labels:
            sentences_without += 1
            continue
        sentences_with += 1

        for label in labels:
            if label not in cited:
                cited.append(label)
            if label in provided:
                if label not in resolved:
                    resolved.append(label)
            else:
                if label not in unresolved:
                    unresolved.append(label)
                issues.append(
                    CitationIssue(
                        kind="unresolved_label",
                        label=label,
                        detail=f"citation label {label} has no matching context",
                        sentence=sentence[:200],
                    )
                )

        resolved_texts = [text_by_label[label] for label in labels if label in provided]
        if not resolved_texts:
            # Every label in this sentence is fabricated: nothing can be grounded.
            continue
        haystack = "\n".join(resolved_texts)
        haystack_numbers = _numbers_in(haystack)

        sentence_ok = True
        for number in sorted(_numbers_in(sentence)):
            if number not in haystack_numbers:
                sentence_ok = False
                ungrounded_numbers += 1
                issues.append(
                    CitationIssue(
                        kind="ungrounded_number",
                        detail=(
                            f"number {number} in the claim does not appear in the "
                            f"cited context(s) {', '.join(labels)}"
                        ),
                        sentence=sentence[:200],
                    )
                )

        if check_quotes:
            normalized_haystack = re.sub(r"\s+", "", haystack)
            for quote in QUOTED_SPAN_RE.findall(sentence):
                if re.sub(r"\s+", "", quote) not in normalized_haystack:
                    sentence_ok = False
                    ungrounded_quotes += 1
                    issues.append(
                        CitationIssue(
                            kind="ungrounded_quote",
                            detail=(
                                "quoted span does not appear verbatim in the cited "
                                f"context(s) {', '.join(labels)}"
                            ),
                            sentence=sentence[:200],
                        )
                    )

        if _lexically_supported(sentence, haystack):
            lexically_supported += 1
        if sentence_ok:
            grounded += 1

    return CitationIntegrityReport(
        cited_labels=tuple(cited),
        resolved_labels=tuple(resolved),
        unresolved_labels=tuple(unresolved),
        sentences_with_citations=sentences_with,
        sentences_without_citations=sentences_without,
        grounded_sentences=grounded,
        ungrounded_number_count=ungrounded_numbers,
        ungrounded_quote_count=ungrounded_quotes,
        sentences_lexically_supported=lexically_supported,
        issues=tuple(issues),
    )
