import re

from app.schemas.chat import ChatResponse, Citation, SupportingClaim

EVIDENCE_PRIORITY = {"primary": 0, "supporting": 1, "conflicting": 2}
CONFLICT_MARKERS = (
    "存在冲突",
    "说法不一致",
    "来源不一致",
    "无法确定",
    "不能确定",
    "inconsistent",
    "conflict",
)
EVIDENCE_SUMMARY_TITLES = {
    "primary": "Primary evidence",
    "supporting": "Supporting evidence",
    "conflicting": "Conflicting evidence",
}
MAX_SPAN_TEXT_CHARS = 180
SENTENCE_SPLIT_RE = re.compile(r"(?<=[\.\!\?。！？；;])\s+|\n+")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
CJK_SEQUENCE_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]+")
CITATION_LABEL_RE = re.compile(r"\[(C\d+)\]")
CITATION_GROUP_RE = re.compile(r"\[((?:C\d+\s*,\s*)*C\d+)\]")
NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")
FINAL_ANSWER_SECTION_RE = re.compile(
    r"(?:^|\n)(?:#{1,6}\s*)?(?:Final Answer|Answer)\s*:\s*\n?(.*?)(?=\n(?:#{1,6}\s*)?Supporting Claims\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
SUPPORTING_CLAIMS_SECTION_RE = re.compile(
    r"(?:^|\n)(?:#{1,6}\s*)?Supporting Claims\s*:\s*\n?(.*)\Z",
    re.IGNORECASE | re.DOTALL,
)
CLAIM_LINE_RE = re.compile(r"^\s*[-*]\s+(.*\S)\s*$")
CLAIM_TYPE_PREFIX_RE = re.compile(
    r"^\[(factual|conditional|conflict|insufficiency)\]\s*",
    re.IGNORECASE,
)
CONDITIONAL_CLAIM_MARKERS = (
    "if ",
    "when ",
    "unless ",
    "subject to ",
    "provided that ",
    "under ",
    "depending on ",
)
CONFLICT_CLAIM_MARKERS = (
    "conflict",
    "inconsistent",
    "disagree",
    "contradict",
    "not aligned",
    "存在冲突",
    "来源不一致",
    "说法不一致",
)
INSUFFICIENCY_CLAIM_MARKERS = (
    "insufficient",
    "not enough evidence",
    "cannot determine",
    "unable to determine",
    "unclear",
    "unknown",
    "evidence is limited",
    "无法确认",
    "信息不足",
    "证据不足",
    "不能确定",
)


class AnswerFormatter:
    def format(
        self, answer: str, contexts: list[dict[str, object]], trace_id: str | None
    ) -> ChatResponse:
        answer_body, supporting_claims = self._extract_answer_plan(answer.strip())
        ordered_contexts = sorted(
            contexts,
            key=lambda item: (
                EVIDENCE_PRIORITY.get(str(item.get("evidence_role")), 3),
                -float(item.get("score", 0.0) or 0.0),
            ),
        )
        citations_by_chunk: dict[str, Citation] = {}
        citations_by_label: dict[str, Citation] = {}
        context_text_by_label: dict[str, str] = {}
        for context in ordered_contexts:
            chunk_id = str(context["chunk_id"])
            if chunk_id not in citations_by_chunk:
                span_text, span_start, span_end = self._extract_citation_span(
                    answer_body,
                    str(context.get("text", "") or ""),
                )
                citation = Citation(
                    citation_label=str(context.get("citation_label"))
                    if context.get("citation_label") is not None
                    else None,
                    chunk_id=chunk_id,
                    source=str(context["source"]),
                    score=float(context["score"]),
                    evidence_role=str(context.get("evidence_role"))
                    if context.get("evidence_role") is not None
                    else None,
                    wiki_status=str(context.get("wiki_status"))
                    if context.get("wiki_status") is not None
                    else None,
                    span_text=span_text,
                    span_start=span_start,
                    span_end=span_end,
                )
                citations_by_chunk[chunk_id] = citation
                if citation.citation_label:
                    citations_by_label[citation.citation_label] = citation
                    context_text_by_label[citation.citation_label] = str(
                        context.get("text", "") or ""
                    )
        normalized_answer = answer_body
        if self._has_conflicting_evidence(ordered_contexts) and not self._mentions_conflict(
            normalized_answer
        ):
            normalized_answer = (
                f"{normalized_answer}\n\nNote: the available evidence is conflicting, so the conclusion may depend on source or version differences."
            ).strip()
        cited_labels = self._extract_cited_labels(normalized_answer)
        cited_labels = self._filter_cited_labels(
            normalized_answer,
            cited_labels,
            citations_by_label,
            context_text_by_label,
        )
        normalized_answer = self._rewrite_citation_markers(normalized_answer, cited_labels)
        cited_labels = self._extract_cited_labels(normalized_answer)
        ordered_citations = self._ordered_citations(
            citations_by_chunk,
            citations_by_label,
            cited_labels,
        )
        if ordered_citations and not cited_labels:
            first_label = ordered_citations[0].citation_label or ordered_citations[0].chunk_id
            normalized_answer = f"{normalized_answer} [{first_label}]"
        return ChatResponse(
            answer=normalized_answer,
            citations=ordered_citations,
            contexts=ordered_contexts,
            supporting_claims=supporting_claims,
            trace_id=trace_id,
        )

    def _has_conflicting_evidence(self, contexts: list[dict[str, object]]) -> bool:
        return any(
            context.get("evidence_role") == "conflicting"
            or context.get("wiki_status") == "conflicting"
            for context in contexts
        )

    def _mentions_conflict(self, answer: str) -> bool:
        lowered = answer.lower()
        return any(marker in answer or marker in lowered for marker in CONFLICT_MARKERS)

    def _append_evidence_summary(
        self,
        answer: str,
        citations: list[Citation],
    ) -> str:
        if not citations:
            return answer
        if any(title in answer for title in EVIDENCE_SUMMARY_TITLES.values()):
            return answer

        grouped: dict[str, list[str]] = {"primary": [], "supporting": [], "conflicting": []}
        for citation in citations:
            role = citation.evidence_role or "supporting"
            if role not in grouped:
                role = "supporting"
            label = citation.citation_label or citation.chunk_id
            grouped[role].append(f"[{label}]")

        lines = []
        for role in ("primary", "supporting", "conflicting"):
            chunk_refs = grouped[role]
            if not chunk_refs:
                continue
            lines.append(f"{EVIDENCE_SUMMARY_TITLES[role]}: {', '.join(chunk_refs)}")
        if not lines:
            return answer
        return f"{answer}\n\n" + "\n".join(lines)

    def _extract_cited_labels(self, answer: str) -> list[str]:
        seen: set[str] = set()
        ordered: list[str] = []
        for group in CITATION_GROUP_RE.findall(answer):
            for match in [label.strip() for label in group.split(",") if label.strip()]:
                if match in seen:
                    continue
                ordered.append(match)
                seen.add(match)
        return ordered

    def _filter_cited_labels(
        self,
        answer: str,
        cited_labels: list[str],
        citations_by_label: dict[str, Citation],
        context_text_by_label: dict[str, str],
    ) -> list[str]:
        labels = [label for label in cited_labels if label in citations_by_label]
        if len(labels) <= 1:
            return labels

        answer_numbers = {
            self._normalize_number(number) for number in NUMBER_RE.findall(answer)
        }
        if not answer_numbers:
            return labels

        number_supported_labels = [
            label
            for label in labels
            if any(
                number in self._normalize_number_text(context_text_by_label.get(label, ""))
                for number in answer_numbers
            )
        ]
        return number_supported_labels or labels

    def _normalize_number(self, value: str) -> str:
        return value.replace(",", "")

    def _normalize_number_text(self, value: str) -> str:
        return "".join(
            self._normalize_number(match) for match in NUMBER_RE.findall(value)
        )

    def _rewrite_citation_markers(self, answer: str, allowed_labels: list[str]) -> str:
        allowed = set(allowed_labels)

        def replace_group(match: re.Match[str]) -> str:
            labels = [
                label.strip()
                for label in match.group(1).split(",")
                if label.strip() in allowed
            ]
            if not labels:
                return ""
            return "[" + ", ".join(labels) + "]"

        rewritten = CITATION_GROUP_RE.sub(replace_group, answer)
        rewritten = re.sub(r"\s+([。！？；;,.])", r"\1", rewritten)
        rewritten = re.sub(r"[ \t]{2,}", " ", rewritten)
        rewritten = re.sub(r"\n{3,}", "\n\n", rewritten)
        return rewritten.strip()

    def _ordered_citations(
        self,
        citations_by_chunk: dict[str, Citation],
        citations_by_label: dict[str, Citation],
        cited_labels: list[str],
    ) -> list[Citation]:
        if cited_labels:
            ordered: list[Citation] = []
            used_chunks: set[str] = set()
            for label in cited_labels:
                citation = citations_by_label.get(label)
                if citation is None or citation.chunk_id in used_chunks:
                    continue
                ordered.append(citation)
                used_chunks.add(citation.chunk_id)
            return ordered
        citations = list(citations_by_chunk.values())
        return citations[:1]

    def _extract_answer_plan(
        self,
        raw_answer: str,
    ) -> tuple[str, list[SupportingClaim]]:
        answer_text = raw_answer.strip()
        supporting_claims: list[SupportingClaim] = []

        final_answer_match = FINAL_ANSWER_SECTION_RE.search(raw_answer)
        if final_answer_match:
            candidate = final_answer_match.group(1).strip()
            if candidate:
                answer_text = candidate

        claims_match = SUPPORTING_CLAIMS_SECTION_RE.search(raw_answer)
        if claims_match:
            supporting_claims = self._parse_supporting_claims(claims_match.group(1))

        return answer_text, supporting_claims

    def _parse_supporting_claims(self, section: str) -> list[SupportingClaim]:
        claims: list[SupportingClaim] = []
        for raw_line in section.splitlines():
            line_match = CLAIM_LINE_RE.match(raw_line)
            if not line_match:
                continue
            line = line_match.group(1).strip()
            if line.lower() == "none":
                continue
            citation_labels = self._extract_cited_labels(line)
            claim_type, claim_body = self._extract_claim_type(line)
            claim_text = CITATION_LABEL_RE.sub("", claim_body).strip()
            if not claim_text:
                continue
            claims.append(
                SupportingClaim(
                    claim_type=claim_type,
                    text=claim_text,
                    citation_labels=citation_labels,
                )
            )
        return claims

    def _extract_claim_type(self, line: str) -> tuple[str, str]:
        type_match = CLAIM_TYPE_PREFIX_RE.match(line)
        if type_match:
            claim_type = type_match.group(1).lower()
            claim_body = line[type_match.end() :].strip()
            return claim_type, claim_body
        inferred_type = self._infer_claim_type(line)
        return inferred_type, line

    def _infer_claim_type(self, line: str) -> str:
        lowered = line.lower()
        if any(marker in lowered for marker in INSUFFICIENCY_CLAIM_MARKERS):
            return "insufficiency"
        if any(marker in lowered for marker in CONFLICT_CLAIM_MARKERS):
            return "conflict"
        if any(marker in lowered for marker in CONDITIONAL_CLAIM_MARKERS):
            return "conditional"
        return "factual"

    def _extract_citation_span(
        self,
        answer: str,
        context_text: str,
    ) -> tuple[str | None, int | None, int | None]:
        text = context_text.strip()
        if not text:
            return None, None, None
        scoring_answer = CITATION_GROUP_RE.sub("", answer)
        candidates = self._span_candidates(text)
        if not candidates:
            candidates = [(text, 0, len(text), False)]
        answer_tokens = self._tokenize(scoring_answer)
        answer_terms = self._cjk_terms(scoring_answer)
        answer_numbers = {
            self._normalize_number(number) for number in NUMBER_RE.findall(scoring_answer)
        }
        best_candidate, best_start, best_end, _ = candidates[0]
        best_score = float("-inf")
        for candidate, start, end, is_table_row in candidates:
            candidate_tokens = self._tokenize(candidate)
            token_score = len(answer_tokens & candidate_tokens)
            term_score = sum(len(term) for term in answer_terms if term in candidate)
            number_text = self._normalize_number_text(candidate)
            number_score = sum(1 for number in answer_numbers if number in number_text)
            if is_table_row and answer_numbers and number_score == 0:
                continue
            table_bonus = 2 if is_table_row else 0
            score = token_score + term_score * 2 + number_score + table_bonus
            if score > best_score or (
                score == best_score and len(candidate) < len(best_candidate)
            ):
                best_candidate = candidate
                best_start = start
                best_end = end
                best_score = score
        span_text = self._preview(best_candidate, MAX_SPAN_TEXT_CHARS)
        return span_text, best_start, best_end

    def _span_candidates(self, text: str) -> list[tuple[str, int, int, bool]]:
        candidates: list[tuple[str, int, int, bool]] = []
        offset = 0
        for line in text.splitlines(keepends=True):
            stripped = line.strip()
            line_body = line.rstrip("\r\n")
            line_start = offset + len(line) - len(line.lstrip())
            line_end = offset + len(line_body)
            offset += len(line)
            if not self._is_table_content_row(stripped):
                continue
            candidates.append((stripped, line_start, line_end, True))

        search_start = 0
        for sentence in SENTENCE_SPLIT_RE.split(text):
            stripped = sentence.strip()
            if not stripped:
                continue
            if stripped.startswith("|") and stripped.count("|") >= 3:
                continue
            start = text.find(stripped, search_start)
            if start < 0:
                start = text.find(stripped)
            if start < 0:
                start = 0
            end = start + len(stripped)
            search_start = end
            candidates.append((stripped, start, end, False))
        return candidates

    def _is_table_content_row(self, value: str) -> bool:
        if not value.startswith("|") or value.count("|") < 3:
            return False
        cells = [cell.strip() for cell in value.strip("|").split("|")]
        if not any(cells):
            return False
        return not all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)

    def _cjk_terms(self, value: str) -> set[str]:
        terms: set[str] = set()
        for match in CJK_SEQUENCE_RE.finditer(value):
            chars = match.group()
            for size in range(2, min(7, len(chars) + 1)):
                for index in range(0, len(chars) - size + 1):
                    terms.add(chars[index : index + size])
        return terms

    def _tokenize(self, value: str) -> set[str]:
        tokens = {token.lower() for token in TOKEN_RE.findall(value)}
        for match in CJK_SEQUENCE_RE.finditer(value):
            chars = match.group()
            tokens.update(chars)
            for size in range(2, min(5, len(chars) + 1)):
                for index in range(0, len(chars) - size + 1):
                    tokens.add(chars[index : index + size])
        return tokens

    def _preview(self, text: str, limit: int) -> str:
        normalized = " ".join(text.split())
        if len(normalized) <= limit:
            return normalized
        return normalized[: limit - 3].rstrip() + "..."
