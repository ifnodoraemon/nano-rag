"""Let the evidence auditor — and the synthesizer — actually see the evidence.

The case that forced this. A document-level corpus (595 real statutes, 23,386
articles) is chunked one document per chunk: "沈阳市民用建筑供热用热管理条例" is
a single 5,676-character chunk, not a section. Retrieve it and the whole
statute is the context; 第二十六条 ("本市正常供热运行期为每年11月1日至翌年3月31日")
sits at character 3,042 of it.

The evidence auditor read each context through ``text[:1600]``. For a
document-level chunk that slice is its title, enactment note and table of
contents — the front matter. So the auditor answered "insufficient: 供暖起止日期
未包含" about a statute whose 第二十六条 states exactly that, and synthesis,
which receives the same verdict, refused as instructed.

Measured over the 270-question end-to-end run: a hard 1,600-character cap per
context exposed 81 of 258 sentences on average (31%). For 16 of the 42 refusals
the sentence matching the auditor's own ``missing_terms`` was present in the
context it had been given — and outside the visible slice.

The fix is not a bigger number. Any fixed prefix has the same shape of bug: the
answer sits at an arbitrary offset, and a longer prefix only moves the cut. So
the budget becomes a per-document allocation, and inside a document the
passages are chosen by the terms the reader is looking for instead of by
position. Nothing here invents content: every passage is a verbatim,
offset-carrying slice of the context, and an excerpt that covers the whole
document reports ``exhausted=True``.

This is deterministic selection — the same kind of stage as BM25. No second
model decides what to read: CLAUDE.md puts ranking in the lexical index and the
reading plan, never in a cross-encoder.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Sentence units for passage selection: a clause split on sentence punctuation,
# or a line. Statutes put one article per unit, so an article stays intact and
# can be shown whole.
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？；!?;])|[\r\n]+")

# Bearings inside the document rather than at its start: an excerpt keeps as
# many of these as fit, so a passage is self-locating and citable.
_ANCHOR_RE = re.compile(
    r"(?:第\s*[0-9〇零一二三四五六七八九十百千]{1,8}\s*[条章节款项])|"
    r"(?<!\d)\d{1,4}(?:\.\d{1,3})+\s*[\u4e00-\u9fff]|"
    r"^\s*(?:[一二三四五六七八九十]{1,2}\s*[、．.]|（[一二三四五六七八九十]{1,2}）)"
)

# Characters of a sentence quoted back for term matching. The sentence is kept
# whole; this only bounds the matching work on a very long paragraph.
_SCORE_WINDOW = 240
# A term match keeps this much text on either side, so a passage carries the
# condition ("自竣工验收之日起") that gives the matched clause its meaning.
_PASSAGE_LEAD = 60
_PASSAGE_TRAIL = 160
# How far before the first match a clipped window starts, the smallest clip
# worth rendering, and the punctuation a clip may end on.
_WINDOW_LEAD = 40
_MIN_CLIP = 48
# How many spread-out windows a document gets when whole units would only cover
# a small part of it, and how large a window is before whole units win.
_MAX_WINDOWS = 4
_MAX_WHOLE_UNITS = 2
# Below this many whole articles the budget is better spent on spread windows.
_MIN_WHOLE_UNITS = 1
_MAX_WINDOW_CHARS = 1200
# A whole unit is only worth keeping when the budget affords a few full ones;
# below this share per unit, spread-out windows carry more of the document.
_MAX_WHOLE_UNIT_SHARE = 400
# The smallest leftover that is worth handing to spread-out windows.
_MIN_WINDOW_BUDGET = 400
# A window narrower than this shows too little of its article to be evidence.
_MIN_GOOD_WINDOW = 400
# The smallest a single window may become when the budget is divided thinly.
_MIN_WINDOW_CHARS = 400
# One ranking entry never takes more than this out of the reader's attention.
_MAX_GATE_CHARS = 600
_CLAUSE_BOUNDARIES = "。；！？，、：;!,:"


@dataclass(frozen=True)
class EvidencePassage:
    """One verbatim slice of a context, with the offset that locates it.

    ``ellipsis_before`` / ``ellipsis_after`` are true when the slice is not at
    the start / end of the context, so a reader can tell coverage continues
    instead of mistaking the slice for the whole document.
    """

    text: str
    offset: int
    ellipsis_before: bool
    ellipsis_after: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "text": self.text,
            "offset": self.offset,
            "length": len(self.text),
            "ellipsis_before": self.ellipsis_before,
            "ellipsis_after": self.ellipsis_after,
        }


@dataclass(frozen=True)
class EvidenceExcerpt:
    """What one context contributes to a character budget."""

    label: str | None
    node_id: str | None
    title: str | None
    source: str | None
    total_chars: int
    passages: list[EvidencePassage] = field(default_factory=list)
    # True when the passages cover the whole context, so "the answer is not in
    # this document" is a conclusion the reader may draw from what it saw.
    exhausted: bool = False

    @property
    def visible_chars(self) -> int:
        return sum(len(passage.text) for passage in self.passages)

    def render(self) -> str:
        """Render for a model prompt, marking elided regions explicitly.

        The markers sit inside the document's own text block and say only where
        the excerpt is cut — never what to conclude. The auditor judges; this
        module only reports what was shown.
        """
        if self.exhausted:
            return "".join(passage.text for passage in self.passages)
        parts: list[str] = []
        for passage in self.passages:
            if passage.ellipsis_before:
                parts.append("[...]")
            parts.append(passage.text)
            if passage.ellipsis_after:
                parts.append("[...]")
        return "".join(parts)

    def as_dict_for_prompt(self) -> dict[str, object]:
        """Serialise for the auditor: text plus where in the document it came from."""
        if not self.passages:
            region = (0, 0)
        else:
            last = self.passages[-1]
            region = (self.passages[0].offset, last.offset + len(last.text))
        return {
            "label": self.label,
            "node_id": self.node_id,
            "title": self.title,
            "source": self.source,
            "document_chars": self.total_chars,
            "visible_region_start": region[0],
            "visible_region_end": region[1],
            "visible_chars": self.visible_chars,
            "excerpt_exhausted": self.exhausted,
            "text": self.render(),
        }


def split_sentences(text: str) -> list[tuple[int, int]]:
    """Split on sentence punctuation and line breaks, keeping exact offsets."""
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_SPLIT.finditer(text):
        end = match.start()
        if end > start:
            spans.append((start, end))
        start = max(start, match.end())
    if start < len(text):
        spans.append((start, len(text)))
    cleaned: list[tuple[int, int]] = []
    for begin, end in spans:
        while begin < end and text[begin].isspace():
            begin += 1
        while end > begin and text[end - 1].isspace():
            end -= 1
        if end > begin:
            cleaned.append((begin, end))
    return cleaned


def _normalize_terms(terms: list[str]) -> list[str]:
    """Trim and dedupe (case-insensitively), keeping the caller's order."""
    seen: set[str] = set()
    out: list[str] = []
    for term in terms:
        normalized = str(term or "").strip()
        key = normalized.casefold()
        if not normalized or key in seen:
            continue
        seen.add(key)
        out.append(normalized)
    return out


def _articles(
    spans: list[tuple[int, int]], anchors: list[tuple[int, int]]
) -> list[tuple[int, int] | None]:
    """For each span, the numbered article it belongs to (or None).

    An article runs from its label to the start of the next label, so the label,
    the article's clauses and its sibling enumerations all map to one range.
    Siblings inside an article ("（一）... （二）...") are labels too, but they do
    not end the article they belong to - the next TOP-LEVEL label does, and a
    top-level label is one that is not indented and is not a parenthesised
    enumeration.
    """
    anchor_set = set(anchors)
    starts = [index for index, span in enumerate(spans) if span in anchor_set]
    result: list[tuple[int, int] | None] = [None] * len(spans)
    for position, start_index in enumerate(starts):
        begin = spans[start_index][0]
        end = (
            spans[starts[position + 1]][0]
            if position + 1 < len(starts)
            else spans[-1][1]
        )
        for index in range(start_index, len(spans)):
            if spans[index][0] >= end:
                break
            result[index] = (begin, end)
    return result


def _last_before(offsets: list[int], limit: int) -> int | None:
    """Largest offset <= ``limit`` in an ascending list, or None."""
    found: int | None = None
    for offset in offsets:
        if offset > limit:
            break
        found = offset
    return found


def _last_clause_boundary(text: str, start: int, stop: int) -> int | None:
    """Last clause boundary in ``text[start:stop]`` as an exclusive end offset."""
    for index in range(stop - 1, start, -1):
        if text[index - 1] in _CLAUSE_BOUNDARIES:
            return index
    return None


def _fit_ranges(
    ranges: list[tuple[int, int]], budget: int
) -> list[tuple[int, int]]:
    """Trim ranges to fit ``budget``, sharing the cut evenly across them.

    Dropping a whole range when only its tail overflows would concentrate the
    excerpt in one place, which is the sampling failure this module exists to
    fix, so each range gives up (about) the same amount instead.
    """
    if budget <= 0:
        return []
    total = sum(end - begin for begin, end in ranges)
    if total <= budget:
        return list(ranges)
    ratio = budget / total
    fitted = [
        [begin, begin + max(1, int((end - begin) * ratio))]
        for begin, end in ranges
    ]
    # Integer rounding must not leave budget on the table (or overspend it).
    remainder = budget - sum(end - begin for begin, end in fitted)
    for item in fitted:
        if remainder <= 0:
            break
        item[1] += 1
        remainder -= 1
    return _merge_ranges([(begin, end) for begin, end in fitted])


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Sort and join overlapping or touching ranges, dropping empty ones."""
    ordered = sorted((begin, end) for begin, end in ranges if end > begin)
    merged: list[tuple[int, int]] = []
    for begin, end in ordered:
        if merged and begin <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((begin, end))
    return merged


@dataclass
class _Entry:
    """One context, analysed against the terms the reader is looking for."""

    label: str | None
    node_id: str | None
    title: str | None
    source: str | None
    text: str
    spans: list[tuple[int, int]]
    anchors: list[tuple[int, int]]
    # For each span: the numbered article holding it, or None.
    articles: list[tuple[int, int] | None]
    # For each span: (distinct terms matched, first match offset, last match
    # offset), or None when the span states none of the reader's terms.
    hits: list[tuple[int, int, int] | None]
    weight: int
    terms: list[str]
    # Start offsets of every sentence unit, ascending: the clause boundaries a
    # window may begin on.
    sentence_starts: list[int]


class EvidenceDriller:
    """Choose which passages of each context a reader gets to see.

    Parameters
    ----------
    max_contexts:
        How many contexts the reader is shown at all.
    total_chars:
        The reader's whole evidence budget.
    max_per_context:
        A ceiling on one context's share, so a single huge statute cannot take
        the entire prompt.
    min_per_context:
        The floor a low-ranked context keeps even when it matches no term, so
        the reader can still learn that the document is unrelated.
    """

    def __init__(
        self,
        *,
        max_contexts: int = 12,
        total_chars: int = 36000,
        max_per_context: int = 6000,
        min_per_context: int = 1200,
    ) -> None:
        if max_contexts < 1:
            raise ValueError("max_contexts must be >= 1")
        if total_chars < 1:
            raise ValueError("total_chars must be >= 1")
        if max_per_context < 1:
            raise ValueError("max_per_context must be >= 1")
        if min_per_context < 0:
            raise ValueError("min_per_context must be >= 0")
        self.max_contexts = max_contexts
        self.total_chars = total_chars
        self.max_per_context = max_per_context
        self.min_per_context = min_per_context

    # -- public ------------------------------------------------------------

    def excerpt(
        self,
        contexts: list[dict[str, object]],
        *,
        watch_terms: list[str],
        budget: int | None = None,
    ) -> list[EvidenceExcerpt]:
        """Allocate ``budget`` characters across ``contexts`` and pick passages.

        ``watch_terms`` are what the reader is looking for — the user's question
        and, on a re-audit, the terms the previous pass said it could not find.
        A context sharing no term with them still gets its floor; it is never
        dropped here (dropping is the auditor's or the reading plan's call, and
        a reader that cannot see a context cannot report on it honestly).
        """
        usable = self.total_chars if budget is None else min(budget, self.total_chars)
        selected = contexts[: self.max_contexts]
        if not selected:
            return []
        terms = _normalize_terms(watch_terms)
        entries = [self._analyse(context, terms) for context in selected]
        shares = self._allocate(entries, usable)
        return [
            self._pick(entry, share) for entry, share in zip(entries, shares, strict=True)
        ]

    def visible_ratio(self, excerpts: list[EvidenceExcerpt]) -> float:
        total = sum(excerpt.total_chars for excerpt in excerpts)
        if total <= 0:
            return 0.0
        visible = sum(excerpt.visible_chars for excerpt in excerpts)
        return round(visible / total, 4)

    # -- analysis ----------------------------------------------------------

    def _analyse(self, context: dict[str, object], terms: list[str]) -> _Entry:
        text = str(context.get("text") or "")
        spans = split_sentences(text)
        anchors = [
            (begin, end) for begin, end in spans if _ANCHOR_RE.match(text, begin, end)
        ]
        hits: list[tuple[int, int, int] | None] = []
        weight = 0
        for begin, end in spans:
            window_end = min(end, begin + _SCORE_WINDOW)
            window = text[begin:window_end].casefold()
            distinct = 0
            first = -1
            last = -1
            for term in terms:
                position = window.find(term.casefold())
                if position < 0:
                    continue
                distinct += 1
                first = position if first < 0 else min(first, position)
                last = max(last, position + len(term))
            weight += distinct
            hits.append((distinct, begin + first, begin + last) if distinct else None)
        return _Entry(
            label=_optional_str(context.get("citation_label")),
            node_id=_optional_str(context.get("node_id") or context.get("chunk_id")),
            title=_optional_str(context.get("title")),
            source=_optional_str(context.get("source") or context.get("source_path")),
            text=text,
            spans=spans,
            anchors=anchors,
            articles=_articles(spans, anchors),
            hits=hits,
            weight=weight,
            terms=terms,
            sentence_starts=[begin for begin, _end in spans],
        )

    # -- budget ------------------------------------------------------------

    def _allocate(self, entries: list[_Entry], usable: int) -> list[int]:
        """Water-fill ``usable`` characters over documents by relevance and rank.

        Every document first gets its floor; the rest is allocated by weight,
        each top-up capped by its own ceiling. Largest-remainder distribution is
        used rather than proportional integer division so the shares sum to
        exactly the budget: rounding loss in an earlier version left every
        excerpt showing only its opening lines.

        A document that matches the question's terms is the one to read
        closely; a document that matches nothing only needs to be recognisable.
        """
        n = len(entries)
        if n == 0:
            return []
        floor = min(self.min_per_context, usable // n)
        shares = [floor] * n
        remaining = usable - floor * n
        if remaining <= 0:
            return shares
        weights = [1 + entry.weight for entry in entries]
        total_weight = sum(weights)
        exact = [remaining * weight / total_weight for weight in weights]
        wants = [int(value) for value in exact]
        leftover = remaining - sum(wants)
        order = sorted(range(n), key=lambda i: exact[i] - wants[i], reverse=True)
        for i in order[:leftover]:
            wants[i] += 1
        for _ in range(n + 1):
            spare = 0
            eligible: list[int] = []
            for i in range(n):
                room = max(0, self.max_per_context - shares[i])
                if wants[i] > room:
                    spare += wants[i] - room
                    wants[i] = room
                if shares[i] + wants[i] < self.max_per_context:
                    eligible.append(i)
            if spare <= 0 or not eligible:
                break
            share_each, extra = divmod(spare, len(eligible))
            if share_each == 0 and extra == 0:
                break
            for position, i in enumerate(eligible):
                wants[i] += share_each + (1 if position < extra else 0)
        return [shares[i] + wants[i] for i in range(n)]

    # -- passage selection -------------------------------------------------

    def _pick(self, entry: _Entry, share: int) -> EvidenceExcerpt:
        text = entry.text
        if not text:
            return self._excerpt(entry, [], exhausted=True)
        if len(text) <= share:
            return self._excerpt(
                entry,
                [
                    EvidencePassage(
                        text=text, offset=0, ellipsis_before=False, ellipsis_after=False
                    )
                ],
                exhausted=True,
            )
        if entry.weight <= 0:
            # Nothing in this document speaks to the question: no passage is
            # better than any other, so it contributes its opening (enough to
            # recognise what it is) and leaves the budget to documents that can
            # answer. Scoring every phrase of an unrelated 40-article law let
            # its tail outrank the article in the document that did match.
            return self._head(entry, share)
        ranges = self._select_ranges(entry, share)
        if not ranges:
            return self._head(entry, share)
        passages = self._passages(entry, ranges)
        # Exhausted only when nothing was elided: the excerpts run from the
        # document's first character to its last with no gap, so "absent from
        # this document" is a conclusion the reader may draw.
        exhausted = (
            bool(passages) and not passages[0].ellipsis_before
            and not any(passage.ellipsis_after for passage in passages)
        )
        return self._excerpt(entry, passages, exhausted=exhausted)

    def _whole_unit_budget(self, entry: _Entry, share: int) -> list[tuple[int, int]]:
        """Units shown in full, continuing forward from the top of the ranking.

        A numbered article is a self-contained normative unit, so it is the
        passage a reader answers from; cutting it can invert it. Only the top of
        the ranking is worth this: after a few articles, the next ones share the
        same vocabulary and say the same kind of thing.
        """
        ranked = self._ranked_matches(entry)
        units: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        used = 0
        for index in ranked:
            span = self._passage_span(entry, index)
            if span in seen:
                continue
            seen.add(span)
            length = span[1] - span[0]
            if used + length > share:
                break
            units.append(span)
            used += length
        return _merge_ranges(units)

    def _ranked_matches(self, entry: _Entry) -> list[int]:
        """Span indices holding a term match, most distinct terms first."""
        return sorted(
            (index for index, hit in enumerate(entry.hits) if hit is not None),
            key=lambda index: (-entry.hits[index][0], entry.spans[index][0]),  # type: ignore[index]
        )

    def _select_ranges(self, entry: _Entry, share: int) -> list[tuple[int, int]]:
        """Pick verbatim ranges under ``share`` characters.

        Two shapes of excerpt, because two things go wrong:

        * The head of the ranking, as whole articles or passages. This is what
          a reader answers from, and it keeps an article intact.
        * Spread-out windows. In a statute every article shares the question's
          vocabulary, so consecutive clauses are the least informative sample
          available and the article that answers the question can be anywhere.
          Half the budget therefore goes to windows spread across the document
          rather than to a longer run of adjacent articles.

        A document that states none of the reader's terms gets its opening and
        gives the rest of its budget back: an arbitrary sample of an unrelated
        document buys nothing.
        """
        units = self._units(entry)
        if not units:
            return self._opening(entry, share)
        if len(units) <= _MAX_WHOLE_UNITS:
            # A couple of matches are the document's own evidence; show them
            # whole, and let the rest of the budget widen those passages rather
            # than sample an unrelated part of the document.
            return _merge_ranges(
                [self._widen(entry, begin, end, share) for begin, end in units]
            )
        # Half the budget is reserved for windows spread through the document;
        # the head may spend the other half, and whatever the head cannot use
        # is handed to the windows. Showing only the head of the ranking was
        # the original defect in a smaller form: in a statute every article
        # shares the question's vocabulary, so the head is arbitrary and the
        # answering article can be at the far end.
        # Windows are what the head cannot be: a sample of the rest of the
        # document. They get half the budget by right, plus whatever the head
        # cannot use, and the head only keeps what is left after them.
        head_share = share // 2
        kept = self._whole_unit_budget(entry, head_share)
        used = sum(end - begin for begin, end in kept)
        # The spread is sized from its reserved half; unused head budget is
        # only offered when it can actually buy another window, so the total
        # never silently falls short.
        spread_budget = share - min(used, head_share)
        spread = self._windows(entry, units, spread_budget)
        if not spread:
            return _merge_ranges(self._whole_unit_budget(entry, share))
        if len(spread) == 1:
            # The head did not need its whole half: spend it on more sample
            # points. Two narrower windows still read the document in two
            # places, which one wide window cannot.
            windows = [
                self._collapse(units, share - used, count)
                for count in (2, 3, _MAX_WINDOWS)
            ]
            spread = next(
                (candidate for candidate in windows if len(candidate) > len(spread)),
                spread,
            )
        # The head never pushes past the budget, and the budget is then fully
        # spent: leftover room goes back to the last window's tail rather than
        # being quietly dropped.
        allowance = share - sum(end - begin for begin, end in spread)
        fitted: list[tuple[int, int]] = []
        spent = 0
        for begin, end in kept:
            if spent + (end - begin) > allowance:
                break
            fitted.append((begin, end))
            spent += end - begin
        slack = share - sum(end - begin for begin, end in [*fitted, *spread])
        if slack > 0 and spread:
            begin, end = spread[-1]
            spread[-1] = (begin, min(len(entry.text), end + slack))
        return _merge_ranges([*fitted, *spread])

    @staticmethod
    def _widen(
        entry: _Entry, begin: int, end: int, room: int
    ) -> tuple[int, int]:
        """Grow a passage to ``room`` characters without leaving its article."""
        article = entry.articles[begin] if begin < len(entry.articles) else None
        low = article[0] if article is not None else 0
        high = article[1] if article is not None else len(entry.text)
        short = max(0, min(room, len(entry.text)) - (end - begin))
        left = min(short // 2, begin - low)
        right = min(short - left, high - end)
        return (begin - left, end + right)

    def _units(self, entry: _Entry) -> list[tuple[int, int]]:
        """Distinct passage units holding a term match, best match first."""
        units: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for index in self._ranked_matches(entry):
            span = self._passage_span(entry, index)
            if span in seen:
                continue
            seen.add(span)
            units.append(span)
        return units

    def _opening(self, entry: _Entry, share: int) -> list[tuple[int, int]]:
        """The document's opening, for a document that answers nothing."""
        return [(0, min(len(entry.text), share))] if share > 0 else []

    def _windows(
        self, entry: _Entry, units: list[tuple[int, int]], share: int
    ) -> list[tuple[int, int]]:
        """Windows spread across the document, under ``share`` in total.

        Spacing is the point. In a statute every article shares the question's
        vocabulary, so consecutive clauses are the least informative sample
        available and the article that answers the question can be anywhere.
        The matching units are cut into contiguous regions and each region
        contributes one window from its start.

        Each window is one unit's span, widened to the width the budget allows
        for this many windows, and never widened past the budget: the excess
        would come out of another document's allocation.
        """
        if share <= _MIN_CLIP or not units:
            return []
        count = min(_MAX_WINDOWS, len(units), max(1, share // _MIN_CLIP))
        while count > 2 and share // count < _MIN_WINDOW_CHARS:
            # Fewer windows, each wide enough to carry a clause. A single
            # window is the last resort, not the default: it is better than
            # spending the allocation on nothing.
            count -= 1
        width = min(share // count, _MAX_GATE_CHARS)
        if width < _MIN_CLIP:
            # A budget too small for two clause-sized windows is still a budget:
            # show one window rather than none, so the allocation is not lost.
            width = min(share, _MAX_GATE_CHARS)
            count = 1
        if width < _MIN_CLIP:
            return []
        windows: list[tuple[int, int]] = []
        for index in range(count):
            lower = (len(units) * index) // count
            begin, end = units[min(len(units) - 1, lower)]
            # A window opens at its region's first match and runs forward; it
            # is never shorter than that match's own passage, because a passage
            # cut mid-sentence is what made the old prefix look like absence.
            stop = min(len(entry.text), begin + width)
            if stop - begin >= _MIN_CLIP:
                windows.append((begin, stop))
        return windows

    def _collapse(
        self, units: list[tuple[int, int]], share: int, count: int
    ) -> list[tuple[int, int]]:
        """``count`` equal windows spread over ``units``, under ``share``."""
        count = max(1, min(count, len(units)))
        width = max(1, share // count)
        spans: list[tuple[int, int]] = []
        for index in range(count):
            lower = (len(units) * index) // count
            begin, _end = units[min(len(units) - 1, lower)]
            spans.append((begin, begin + width))
        return spans

    def _has_hit(self, entry: _Entry, begin: int, end: int) -> bool:
        return any(entry.hits[index] for index, span in enumerate(entry.spans) if span[0] >= begin and span[1] <= end)

    def _passage_span(self, entry: _Entry, index: int) -> tuple[int, int]:
        """The article or sentence holding the term match, with its conditions.

        A numbered article is a self-contained normative unit, so cutting it can
        invert it; a plain sentence is widened just enough to keep the condition
        that qualifies the matched clause ("自竣工验收之日起...").
        """
        hit = entry.hits[index]
        begin, end = entry.spans[index]
        if hit is None:
            return begin, end
        _distinct, first, last = hit
        enclosing = entry.articles[index]
        if enclosing is not None:
            return enclosing
        return (max(begin, first - _PASSAGE_LEAD), min(end, last + _PASSAGE_TRAIL))

    def _clip(self, entry: _Entry, index: int, room: int) -> tuple[int, int] | None:
        """A window around the term match when the whole passage will not fit."""
        hit = entry.hits[index]
        if hit is None or room <= _MIN_CLIP:
            return None
        begin, end = entry.spans[index]
        _distinct, first, _last = hit
        start_at = max(begin, first - _WINDOW_LEAD)
        stop = min(end, start_at + room)
        boundary = _last_clause_boundary(entry.text, start_at, stop)
        if boundary is not None:
            stop = boundary
        if stop <= start_at:
            return None
        return (start_at, stop)

    def _head(self, entry: _Entry, share: int) -> EvidenceExcerpt:
        text = entry.text
        if share <= 0:
            return self._excerpt(entry, [], exhausted=False)
        head = text[:share]
        return self._excerpt(
            entry,
            [
                EvidencePassage(
                    text=head,
                    offset=0,
                    ellipsis_before=False,
                    ellipsis_after=len(head) < len(text),
                )
            ],
            exhausted=len(head) >= len(text),
        )

    def _passages(
        self, entry: _Entry, ranges: list[tuple[int, int]]
    ) -> list[EvidencePassage]:
        text = entry.text
        passages: list[EvidencePassage] = []
        for begin, end in _merge_ranges(ranges):
            passages.append(
                EvidencePassage(
                    text=text[begin:end],
                    offset=begin,
                    ellipsis_before=begin > 0,
                    ellipsis_after=end < len(text),
                )
            )
        return passages

    def _excerpt(
        self, entry: _Entry, passages: list[EvidencePassage], *, exhausted: bool
    ) -> EvidenceExcerpt:
        return EvidenceExcerpt(
            label=entry.label,
            node_id=entry.node_id,
            title=entry.title,
            source=entry.source,
            total_chars=len(entry.text),
            passages=passages,
            exhausted=exhausted,
        )


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None
