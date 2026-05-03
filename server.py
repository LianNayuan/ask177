"""HTTP API server for RAG Q&A. Deploy on cloud server and call via POST /ask.

Usage:
  python server.py                        # default: 0.0.0.0:8000
  python server.py --port 8080            # custom port
  python server.py --host 127.0.0.1       # local only
  python server.py --cache index.pkl      # custom cache path

When packaged with PyInstaller, place index.pkl, data.db, and .env next to the .exe.
"""

import sys
from pathlib import Path

# When frozen by PyInstaller, find files next to the .exe, not next to the script
if getattr(sys, "frozen", False):
    APP_DIR = Path(sys.executable).parent
else:
    APP_DIR = Path(__file__).parent

from database import Database
from simple_rag import SimpleRAG, load_dotenv

# ── Config ─────────────────────────────────────────────────────────

_env = load_dotenv(str(APP_DIR / ".env"))
API_KEY = _env.get("DEEPSEEK_API_KEY", "")
CACHE_FILE = str(APP_DIR / "index.pkl")

# ── FastAPI app (lazy init, after argparse) ────────────────────────

app = None  # set by main()
rag: SimpleRAG | None = None
db: Database | None = None


def create_app():
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel

    app = FastAPI(title="RAG Q&A API", version="1.0")

    class QuestionRequest(BaseModel):
        question: str
        session_id: int = 0  # 0 = new session, >0 = continue existing

    class AnswerResponse(BaseModel):
        answer: str
        session_id: int

    @app.get("/health")
    def health():
        return {"status": "ok", "files": len(rag._file_names), "chunks": len(rag._chunks)}

    @app.post("/ask", response_model=AnswerResponse)
    def ask(req: QuestionRequest):
        if not req.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")

        # Session management
        sid = req.session_id
        if sid and db.get_conversation(sid):
            # Continuing an existing session
            pass
        elif req.question.strip() == "/new":
            # Explicit new session request
            sid = db.create_conversation()
            return AnswerResponse(answer=f"New session #{sid} started.", session_id=sid)
        else:
            # First message or invalid session_id → create new
            sid = db.create_conversation()

        # Load history and ask
        rows = db.get_history(sid, limit=10)
        history = [{"role": r["role"], "content": r["content"]} for r in rows]
        answer = rag.ask(req.question, history=history)
        answer = SimpleRAG._sanitize(answer)

        # Persist
        db.add_message(sid, "user", SimpleRAG._sanitize(req.question))
        db.add_message(sid, "assistant", answer)

        return AnswerResponse(answer=answer, session_id=sid)

    return app


# ── Main ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not API_KEY:
        print("Missing DEEPSEEK_API_KEY in .env file.")
        sys.exit(1)

    # Parse args
    host, port = "0.0.0.0", 8000
    cache_file = CACHE_FILE
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--host" and i + 1 < len(args):
            host = args[i + 1]
            i += 2
        elif args[i] == "--port" and i + 1 < len(args):
            port = int(args[i + 1])
            i += 2
        elif args[i] == "--cache" and i + 1 < len(args):
            cache_file = args[i + 1]
            i += 2
        else:
            print(f"Unknown argument: {args[i]}")
            sys.exit(1)

    # Init database (for knowledge metadata, glossary, and query logging)
    db = Database(str(APP_DIR / "data.db"))
    if db.stats()["glossary_entries"] == 0:
        db.import_glossary_from_file(str(APP_DIR / "knowledge/glossary.md"))

    # Load index (force=True skips freshness check — no source .md files needed)
    rag = SimpleRAG(api_key=API_KEY, verbose=False)
    if not rag.load_cache(cache_file, force=True, db=db):
        print(f"Error: {cache_file} not found and no knowledge in data.db.")
        print("Build it locally: python build_tfidf.py")
        print("Then upload index.pkl and data.db to the server.")
        sys.exit(1)

    print(f"Loaded {len(rag._file_names)} files, {len(rag._chunks)} chunks"
          f"  glossary: {len(rag._glossary)} entries")

    # Create app and run
    app = create_app()

    import uvicorn
    print(f"Starting server at http://{host}:{port}")
    print(f"API docs: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port, log_level="info")
