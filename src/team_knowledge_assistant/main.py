from team_knowledge_assistant.config import settings
from fastapi import FastAPI, Response
import psycopg

app = FastAPI(title="Team Knowledge Assistant")

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