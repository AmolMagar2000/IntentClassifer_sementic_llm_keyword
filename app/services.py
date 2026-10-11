from __future__ import annotations

import asyncio
import json
import logging
import re
from functools import lru_cache
from typing import Any

from qdrant_client import QdrantClient, models

from app.config import get_settings
from app.models import LLMDecision

logger = logging.getLogger(__name__)
settings = get_settings()


ROUTER_SYSTEM_PROMPT = """You are an intent classifier for an HR and compliance assistant.

Classify the complete user sentence into exactly one intent:
- sql_agent: asks to count, list, filter, group, compare, or retrieve stored records.
- compliance: asks for an overall compliance snapshot, risk dashboard, alerts, or compliance health.
- rag: asks to explain a policy, rule, definition, procedure, obligation, or required evidence.
- clarify: the request is ambiguous or does not contain enough information.

Important distinctions:
- Do not classify from a single keyword. Interpret the complete sentence.
- "Does absence count toward the reporting threshold?" is rag, because count is used conceptually.
- "Count employees absent this month" is sql_agent, because it requests a database calculation.
- "List expiring visas" is sql_agent; "Explain visa-expiry rules" is rag.
- Use compliance only for dashboard-style summaries or alerts, not individual record lists.
- Confidence must reflect how clearly the sentence belongs to one intent.
- If two intents are equally plausible, return clarify with one short leading question.

Return JSON only:
{
  "intent": "sql_agent|compliance|rag|clarify",
  "confidence": 0.0,
  "reason": "one short reason based on the full sentence",
  "clarification_question": null
}
"""


def normalize_query(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9\s'-]", " ", value)
    return re.sub(r"\s+", " ", value)


@lru_cache
def qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=settings.QDRANT_URL,
        api_key=settings.QDRANT_API_KEY or None,
        timeout=10,
    )


def _embed_sync(texts: list[str]) -> list[list[float]]:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    result = client.models.embed_content(
        model=settings.GEMINI_EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY",
            output_dimensionality=settings.EMBEDDING_DIMENSION,
        ),
    )
    return [list(item.values) for item in result.embeddings]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    return await asyncio.to_thread(_embed_sync, texts)


def load_intents() -> list[dict[str, Any]]:
    return json.loads(settings.intents_file.read_text(encoding="utf-8"))


async def seed_qdrant() -> int:
    """Embed and upsert all example utterances. Safe to call more than once."""
    def _ensure_collection() -> None:
        client = qdrant_client()
        if not client.collection_exists(settings.QDRANT_COLLECTION):
            client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=settings.EMBEDDING_DIMENSION,
                    distance=models.Distance.COSINE,
                ),
            )

    # Confirm Qdrant is ready before spending an embedding API call.
    await asyncio.to_thread(_ensure_collection)

    intent_data = load_intents()
    rows = [
        {"intent": group["intent"], "description": group["description"], "utterance": utterance}
        for group in intent_data
        for utterance in group["utterances"]
    ]
    vectors = await embed_texts([row["utterance"] for row in rows])

    def _seed() -> None:
        client = qdrant_client()
        client.upsert(
            collection_name=settings.QDRANT_COLLECTION,
            points=[
                models.PointStruct(id=index + 1, vector=vector, payload=row)
                for index, (row, vector) in enumerate(zip(rows, vectors, strict=True))
            ],
            wait=True,
        )

    await asyncio.to_thread(_seed)
    logger.info("QDRANT seeded collection=%s utterances=%d", settings.QDRANT_COLLECTION, len(rows))
    return len(rows)


async def semantic_route(query: str) -> dict[str, Any]:
    [vector] = await embed_texts([query])

    def _search():
        return qdrant_client().query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=vector,
            limit=settings.TOP_K,
            with_payload=True,
        ).points

    hits = await asyncio.to_thread(_search)
    matches = [
        {
            "intent": str(hit.payload.get("intent")),
            "utterance": str(hit.payload.get("utterance")),
            "score": round(float(hit.score), 4),
        }
        for hit in hits
    ]
    best = matches[0] if matches else None
    return {"best": best, "matches": matches}


async def _groq_route(query: str) -> LLMDecision:
    if not settings.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not configured")

    from groq import AsyncGroq

    response = await AsyncGroq(api_key=settings.GROQ_API_KEY).chat.completions.create(
        model=settings.GROQ_MODEL,
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ],
        temperature=0,
        max_tokens=300,
        response_format={"type": "json_object"},
    )
    return LLMDecision.model_validate_json(response.choices[0].message.content or "{}")


def _gemini_route_sync(query: str) -> LLMDecision:
    if not settings.GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY is not configured")

    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.GEMINI_API_KEY)
    response = client.models.generate_content(
        model=settings.GEMINI_LLM_MODEL,
        contents=f"{ROUTER_SYSTEM_PROMPT}\n\nUser sentence:\n{query}",
        config=types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=300,
            response_mime_type="application/json",
        ),
    )
    return LLMDecision.model_validate_json(response.text or "{}")


async def llm_route(query: str) -> tuple[LLMDecision, str]:
    errors: list[str] = []
    try:
        return await asyncio.wait_for(_groq_route(query), timeout=12), "groq"
    except Exception as exc:
        errors.append(f"Groq: {type(exc).__name__}")
        logger.warning("LLM router primary failed: %s", exc)

    try:
        decision = await asyncio.wait_for(asyncio.to_thread(_gemini_route_sync, query), timeout=12)
        return decision, "gemini-fallback"
    except Exception as exc:
        errors.append(f"Gemini: {type(exc).__name__}")
        logger.warning("LLM router fallback failed: %s", exc)
        raise RuntimeError("; ".join(errors)) from exc


def keyword_route(query: str) -> dict[str, Any] | None:
    """Last fallback. Requires multiple signals; never routes on one word."""
    text = normalize_query(query)

    action_terms = ("count", "how many", "list", "show", "find", "total", "average", "group")
    data_terms = ("employee", "attendance", "leave", "visa", "payroll", "salary", "hours", "department", "record")
    policy_terms = ("explain", "policy", "rule", "requirement", "required", "evidence", "what does", "how should", "must")
    compliance_summary_terms = ("overall", "snapshot", "dashboard", "health", "urgent alerts", "risk summary")
    compliance_terms = ("compliance", "rtw", "right to work", "visa", "document", "risk", "alert")

    has_action = any(term in text for term in action_terms)
    has_data = any(term in text for term in data_terms)
    has_policy = any(term in text for term in policy_terms)
    has_summary = any(term in text for term in compliance_summary_terms)
    has_compliance = any(term in text for term in compliance_terms)

    if has_policy and has_compliance:
        return {"intent": "rag", "confidence": 0.72, "reason": "policy phrase + compliance subject"}
    if has_summary and has_compliance:
        return {"intent": "compliance", "confidence": 0.72, "reason": "summary phrase + compliance subject"}
    if has_action and has_data and not has_policy:
        return {"intent": "sql_agent", "confidence": 0.72, "reason": "data operation + business entity"}
    return None
