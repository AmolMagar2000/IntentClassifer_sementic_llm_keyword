from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import services
from app.config import get_settings
from app.graph import classifier_graph
from app.models import ClassifyRequest, ClassifyResponse

settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI):
    # try:
    #     count = await services.seed_qdrant()
    #     logger.info("Semantic router ready with %d utterances", count)
    # except Exception as exc:
    #     # The API stays available. Graph fallback continues with LLM and rules.
    #     logger.warning("Qdrant seed skipped: %s", exc)
    yield


app = FastAPI(title=settings.APP_NAME, version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
)
app.mount("/static", StaticFiles(directory=settings.frontend_dir), name="static")


@app.get("/", include_in_schema=False)
async def home():
    return FileResponse(settings.frontend_dir / "index.html")


@app.get("/api/health")
async def health():
    return {"status": "ok", "semantic_threshold": settings.SEMANTIC_THRESHOLD}


@app.get("/api/intents")
async def intents():
    return services.load_intents()


@app.post("/api/reseed")
async def reseed():
    try:
        return {"seeded": await services.seed_qdrant()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/classify", response_model=ClassifyResponse)
async def classify(request: ClassifyRequest):
    initial_state = {
        "query": request.query.strip(),
        "logs": [f"[start] query={request.query.strip()!r}"],
        "execution_path": ["START"],
        "top_matches": [],
    }
    result = await classifier_graph.ainvoke(initial_state)
    result["execution_path"] = [*result.get("execution_path", []), "END"]
    return ClassifyResponse(**result)
