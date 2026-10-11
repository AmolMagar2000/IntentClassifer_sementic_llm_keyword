from __future__ import annotations

import logging
from typing import Literal

from langgraph.graph import END, START, StateGraph

from app import services
from app.config import get_settings
from app.models import GraphState

logger = logging.getLogger(__name__)
settings = get_settings()


def _append(state: GraphState, node: str, message: str) -> dict:
    logs = [*state.get("logs", []), message]
    path = [*state.get("execution_path", []), node]
    logger.info(message)
    return {"logs": logs, "execution_path": path}


async def semantic_router_node(state: GraphState) -> GraphState:
    update = _append(state, "semantic_router", "[semantic] generating query embedding")
    try:
        result = await services.semantic_route(state["query"])
        best = result["best"]
        score = float(best["score"]) if best else 0.0
        message = (
            f"[semantic] best_intent={best['intent'] if best else 'none'} "
            f"score={score:.4f} threshold={settings.SEMANTIC_THRESHOLD:.2f}"
        )
        update["logs"].append(message)
        logger.info(message)
        update["top_matches"] = result["matches"]
        if best and score >= settings.SEMANTIC_THRESHOLD:
            update.update(
                intent=best["intent"],
                confidence=score,
                route_source="semantic",
                reason="Query is semantically similar to a stored example utterance.",
                matched_utterance=best["utterance"],
            )
    except Exception as exc:
        message = f"[semantic] unavailable={type(exc).__name__}; falling back to LLM"
        update["logs"].append(message)
        logger.warning(message)
    return {**state, **update}


def after_semantic(state: GraphState) -> Literal["llm_router", "finalize"]:
    return "finalize" if state.get("route_source") == "semantic" else "llm_router"


async def llm_router_node(state: GraphState) -> GraphState:
    update = _append(state, "llm_router", "[llm] semantic route was uncertain; calling structured LLM router")
    try:
        decision, provider = await services.llm_route(state["query"])
        message = (
            f"[llm] provider={provider} intent={decision.intent} "
            f"confidence={decision.confidence:.4f} threshold={settings.LLM_THRESHOLD:.2f}"
        )
        update["logs"].append(message)
        logger.info(message)
        if decision.confidence >= settings.LLM_THRESHOLD:
            update.update(
                intent=decision.intent,
                confidence=decision.confidence,
                route_source="llm",
                reason=decision.reason,
                clarification_question=decision.clarification_question,
            )
    except Exception as exc:
        message = f"[llm] unavailable={type(exc).__name__}; falling back to keyword rules"
        update["logs"].append(message)
        logger.warning(message)
    return {**state, **update}


def after_llm(state: GraphState) -> Literal["keyword_router", "clarify", "finalize"]:
    if state.get("route_source") != "llm":
        return "keyword_router"
    if state.get("intent") == "clarify":
        return "clarify"
    return "finalize"


async def keyword_router_node(state: GraphState) -> GraphState:
    update = _append(state, "keyword_router", "[keyword] applying conservative multi-signal fallback rules")
    decision = services.keyword_route(state["query"])
    if decision:
        message = f"[keyword] intent={decision['intent']} confidence={decision['confidence']:.2f}"
        update["logs"].append(message)
        logger.info(message)
        update.update(
            intent=decision["intent"],
            confidence=decision["confidence"],
            route_source="keyword",
            reason=decision["reason"],
        )
    else:
        update["logs"].append("[keyword] no safe multi-signal match; clarification required")
    return {**state, **update}


def after_keyword(state: GraphState) -> Literal["clarify", "finalize"]:
    return "finalize" if state.get("route_source") == "keyword" else "clarify"


async def clarify_node(state: GraphState) -> GraphState:
    update = _append(state, "clarify", "[clarify] all classifiers were uncertain")
    return {
        **state,
        **update,
        "intent": "clarify",
        "confidence": 0.0,
        "route_source": "clarify",
        "reason": "The request could not be classified safely.",
        "clarification_question": (
            "Do you want stored HR data, an overall compliance summary, "
            "or an explanation of a policy?"
        ),
    }


async def finalize_node(state: GraphState) -> GraphState:
    update = _append(
        state,
        "finalize",
        f"[final] intent={state.get('intent')} source={state.get('route_source')} confidence={state.get('confidence', 0):.4f}",
    )
    return {**state, **update}


builder = StateGraph(GraphState)
builder.add_node("semantic_router", semantic_router_node)
builder.add_node("llm_router", llm_router_node)
builder.add_node("keyword_router", keyword_router_node)
builder.add_node("clarify", clarify_node)
builder.add_node("finalize", finalize_node)

builder.add_edge(START, "semantic_router")
builder.add_conditional_edges("semantic_router", after_semantic)
builder.add_conditional_edges("llm_router", after_llm)
builder.add_conditional_edges("keyword_router", after_keyword)
builder.add_edge("clarify", "finalize")
builder.add_edge("finalize", END)

classifier_graph = builder.compile()
