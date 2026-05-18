import pytest

from app.retrieval.evidence_planner import EvidencePlanner, EvidencePlannerConfig
from app.retrieval.query_router import QueryRoute


class FakePlannerGenerationClient:
    async def generate(self, messages):  # noqa: ANN001, ARG002
        return {
            "content": (
                '{"answer_strategy":"conflict",'
                '"primary_evidence":["C1"],'
                '"conditions":["C1 only applies to low vitamin D adults"],'
                '"relations":[{"source":"C1","target":"C2","relation":"contradicts","reason":"trial disagrees"}],'
                '"context_annotations":{"C1":{"claim_role":"condition","claim_scope":"low vitamin D adults",'
                '"certainty":"weak","discourse_units":[{"role":"condition","text":"low vitamin D adults"},'
                '{"role":"conclusion","text":"flu incidence decreased 12%"}]},'
                '"C2":{"claim_role":"conflict","certainty":"strong","discourse_units":[{"role":"exception","text":"no significant association"}]}},'
                '"outline":["compare scoped finding","explain conflicting trial","give qualified answer"]}'
            )
        }


class BrokenPlannerGenerationClient:
    async def generate(self, messages):  # noqa: ANN001, ARG002
        raise RuntimeError("planner down")


def _contexts() -> list[dict[str, object]]:
    return [
        {
            "citation_label": "C1",
            "chunk_id": "a",
            "source": "paper-a",
            "score": 0.9,
            "text": "In winter adults with low vitamin D, supplementation reduced flu incidence by 12%.",
            "evidence_role": "primary",
        },
        {
            "citation_label": "C2",
            "chunk_id": "b",
            "source": "paper-b",
            "score": 0.8,
            "text": "A large randomized trial found no statistically significant association.",
            "evidence_role": "conflicting",
        },
    ]


@pytest.mark.asyncio
async def test_evidence_planner_parses_llm_discourse_plan() -> None:
    planner = EvidencePlanner(
        generation_client=FakePlannerGenerationClient(),
        config=EvidencePlannerConfig(enabled=True),
    )

    plan = await planner.plan(
        "Can vitamin D prevent flu?",
        _contexts(),
        QueryRoute(route="conflict"),
    )
    annotated = planner.annotate_contexts(_contexts(), plan)

    assert plan["answer_strategy"] == "conflict"
    assert plan["primary_evidence"] == ["C1"]
    assert plan["relations"][0]["relation"] == "contradicts"
    assert annotated[0]["claim_role"] == "condition"
    assert annotated[1]["claim_role"] == "conflict"


@pytest.mark.asyncio
async def test_evidence_planner_degrades_to_metadata_only_plan() -> None:
    planner = EvidencePlanner(
        generation_client=BrokenPlannerGenerationClient(),
        config=EvidencePlannerConfig(enabled=True),
    )

    plan = await planner.plan(
        "Can vitamin D prevent flu?",
        _contexts(),
        QueryRoute(route="conflict"),
    )

    assert plan["status"] == "degraded"
    assert plan["answer_strategy"] == "conflict"
    assert plan["relations"][0]["relation"] == "contradicts"
