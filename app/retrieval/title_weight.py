"""Decide how much a title field is worth from the corpus itself.

Why this cannot be a constant. The same title weight helps one corpus and hurts another:

    US Code        title_weight 0.0 -> hit@1 0.3622
                   title_weight 0.3 -> hit@1 0.3878   (+2.6pt)
                   title_weight 0.6 -> hit@1 0.3724

    Chinese laws   title_weight 0.3 -> hit@1 -2.3 to -5.3pt against 0.0

The reason is structural, not statistical noise. A Chinese regulation is titled after its
subject ("重庆市国防动员条例"), so a title match is evidence. A US Code group is titled
after its classification ("BANKS AND BANKING > CHECK TRUNCATION") while the body holds the
actual rule, so a title match is evidence about the taxonomy and not about the answer.

A constant therefore encodes a guess about the corpus. This module measures the thing that
distinguishes the two cases and derives the weight from it:

    title diversity = distinct leading segments / documents

    Chinese laws   595 documents, 595 distinct leading segments -> 1.00   (titles tell us)
    US Code        400 documents,  18 distinct leading segments -> 0.045  (titles repeat)

A title that is unique per document carries information; a title shared by eighty documents
carries almost none. The measure is computed from the documents themselves, in any
language, and encodes no vocabulary - "BANKS AND BANKING" would be just as distinctive as
"重庆市国防动员条例" if each document had its own.

The mapping from diversity to weight is linear and bounded: at full diversity the title is
worth the documented maximum, at no diversity it is worth nothing. It is deliberately a
small, explainable function rather than a fitted curve - there are two data points, and
fitting more parameters to two points is how a generality claim becomes an overfit one.
"""

from __future__ import annotations

import logging
import re
from collections import Counter
from typing import Iterable

logger = logging.getLogger(__name__)

# Weight applied when every document's title is distinctive. Chosen because it is the
# value measured to help on the corpus where titles are informative (US Code, +2.6pt at
# 0.3) without exceeding what that measurement supports.
MAX_TITLE_WEIGHT = 0.3

# Below this diversity the title is treated as carrying no usable signal.
MIN_USABLE_DIVERSITY = 0.2

# A leading segment is the part of a title before a structural separator: " > " for a
# hierarchy, " — " / " - " for a prefix label, or the whole title when there is none. This
# is a separator convention shared by most cataloguing schemes, not a language rule.
_SEGMENT_RE = re.compile(r"\s*(?:>|—|–|-{2,}|\|)\s*")
_TRAILING_NOISE = re.compile(r"[\s\.…]+$")


def leading_segment(title: str) -> str:
    """The most general part of a title, used to detect repeated taxonomy labels."""
    head = _SEGMENT_RE.split(title.strip(), maxsplit=1)[0]
    cleaned = _TRAILING_NOISE.sub("", head).strip().casefold()
    return cleaned or title.strip().casefold()


def title_diversity(titles: Iterable[str]) -> float | None:
    """Distinct leading segments per document, or None when there is nothing to measure.

    ``None`` rather than 0.0: an empty corpus says nothing about title usefulness, and
    treating silence as "no signal" would silently disable the field.
    """
    segments: list[str] = []
    for title in titles:
        text = (title or "").strip()
        if text:
            segments.append(leading_segment(text))
    if not segments:
        return None
    return len(set(segments)) / len(segments)


def weight_for_diversity(diversity: float | None) -> float:
    """Title weight implied by a diversity measurement in [0, 1]."""
    if diversity is None:
        return 0.0
    if diversity <= MIN_USABLE_DIVERSITY:
        return 0.0
    if diversity >= 1.0:
        return MAX_TITLE_WEIGHT
    span = 1.0 - MIN_USABLE_DIVERSITY
    scaled = (diversity - MIN_USABLE_DIVERSITY) / span
    return round(MAX_TITLE_WEIGHT * scaled, 4)


def resolve_title_weight(
    titles: Iterable[str], *, override: float | None = None
) -> tuple[float, float | None]:
    """Weight for this corpus and the diversity it was derived from.

    An explicit ``override`` (the operator setting RAG_BM25_TITLE_WEIGHT) always wins: the
    measurement is a default, not an authority.
    """
    diversity = title_diversity(titles)
    if override is not None:
        return override, diversity
    weight = weight_for_diversity(diversity)
    logger.info(
        "title weight auto: diversity=%s -> %s",
        "n/a" if diversity is None else f"{diversity:.3f}",
        weight,
    )
    return weight, diversity


def count_segments(titles: Iterable[str]) -> Counter[str]:
    """Leading-segment histogram, for diagnostics."""
    return Counter(leading_segment(title) for title in titles if (title or "").strip())
