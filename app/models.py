from typing import Literal, TypedDict

from pydantic import BaseModel, Field


Intent = Literal["sql_agent", "compliance", "rag", "clarify"]


class ClassifyRequest(BaseModel):
    query: str = Field(min_length=2, max_length=2000)


class Match(BaseModel):
    intent: str
    utterance: str
    score: float


class ClassifyResponse(BaseModel):
    query: str
    intent: Intent
    confidence: float
    route_source: Literal["semantic", "llm", "keyword", "clarify"]
    reason: str
    matched_utterance: str | None = None
    clarification_question: str | None = None
    top_matches: list[Match] = Field(default_factory=list)
    execution_path: list[str] = Field(default_factory=list)
    logs: list[str] = Field(default_factory=list)


class LLMDecision(BaseModel):
    intent: Literal["sql_agent", "compliance", "rag", "clarify"]
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    clarification_question: str | None = None


class GraphState(TypedDict, total=False):
    query: str
    intent: Intent
    confidence: float
    route_source: str
    reason: str
    matched_utterance: str | None
    clarification_question: str | None
    top_matches: list[dict]
    execution_path: list[str]
    logs: list[str]
