import pytest

from app import graph, services


@pytest.mark.asyncio
async def test_semantic_route_stops_before_llm(monkeypatch):
    async def semantic(_):
        return {
            "best": {"intent": "sql_agent", "utterance": "Count employees", "score": 0.91},
            "matches": [],
        }

    async def should_not_run(_):
        raise AssertionError("LLM should not run for a high semantic score")

    monkeypatch.setattr(services, "semantic_route", semantic)
    monkeypatch.setattr(services, "llm_route", should_not_run)
    result = await graph.classifier_graph.ainvoke({"query": "count staff", "logs": [], "execution_path": []})
    assert result["intent"] == "sql_agent"
    assert result["route_source"] == "semantic"


@pytest.mark.asyncio
async def test_low_semantic_score_uses_llm(monkeypatch):
    async def semantic(_):
        return {"best": {"intent": "sql_agent", "utterance": "Count employees", "score": 0.42}, "matches": []}

    async def llm(_):
        from app.models import LLMDecision
        return LLMDecision(intent="rag", confidence=0.88, reason="Policy explanation"), "mock"

    monkeypatch.setattr(services, "semantic_route", semantic)
    monkeypatch.setattr(services, "llm_route", llm)
    result = await graph.classifier_graph.ainvoke({"query": "explain the RTW rules", "logs": [], "execution_path": []})
    assert result["intent"] == "rag"
    assert result["route_source"] == "llm"


@pytest.mark.asyncio
async def test_provider_failures_use_keyword_fallback(monkeypatch):
    async def no_semantic(_):
        raise RuntimeError("Qdrant unavailable")

    async def no_llm(_):
        raise RuntimeError("LLMs unavailable")

    monkeypatch.setattr(services, "semantic_route", no_semantic)
    monkeypatch.setattr(services, "llm_route", no_llm)
    result = await graph.classifier_graph.ainvoke({"query": "count employees by department", "logs": [], "execution_path": []})
    assert result["intent"] == "sql_agent"
    assert result["route_source"] == "keyword"


@pytest.mark.asyncio
async def test_all_uncertain_goes_to_clarification(monkeypatch):
    async def semantic(_):
        return {"best": {"intent": "rag", "utterance": "Explain policy", "score": 0.30}, "matches": []}

    async def no_llm(_):
        raise RuntimeError("LLMs unavailable")

    monkeypatch.setattr(services, "semantic_route", semantic)
    monkeypatch.setattr(services, "llm_route", no_llm)
    result = await graph.classifier_graph.ainvoke({"query": "help me with this", "logs": [], "execution_path": []})
    assert result["intent"] == "clarify"
    assert result["route_source"] == "clarify"


def test_single_keyword_is_not_enough():
    assert services.keyword_route("count") is None
    assert services.keyword_route("visa") is None

