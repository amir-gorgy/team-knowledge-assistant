from team_knowledge_assistant.answer import Answer, ChatTruncatedError, answer
from team_knowledge_assistant.config import settings
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
import httpx2
import psycopg

app = FastAPI(title="Team Knowledge Assistant")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    k: int = Field(default=5, ge=1, le=20, description="chunks retrieved before the not-found gate")

@app.get("/")
async def root():
    return {"message": "Welcome to Team Knowledge Assistant!"}

@app.get("/health")
def health(response: Response):
    try:
        with psycopg.connect(settings.database_url, connect_timeout=2) as conn:
            row = conn.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
            ).fetchone()
        return {"status": "ok", "db": "ok", "pgvector": row[0] if row else None}
    except psycopg.OperationalError:
        response.status_code = 503
        return {"status": "degraded", "db": "unreachable"}


@app.post("/ask")
def ask(request: AskRequest) -> Answer:
    """Answer a question from the docs with [n] citations, or found=false instead of a guess.

    With ANSWER_BACKEND=copilot no model is called: `text` is null and `sources` holds the numbered
    chunks that passed the not-found gate, for the caller to write the answer from.

    A plain def, not async: retrieval and the model call block, so FastAPI runs this in a threadpool.
    """
    try:
        return answer(request.question, k=request.k)
    except ChatTruncatedError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except (psycopg.OperationalError, httpx2.HTTPError) as e:
        # Postgres, the embeddings server or the chat model is down or returned an error.
        raise HTTPException(status_code=503, detail=f"a backing service failed: {type(e).__name__}")