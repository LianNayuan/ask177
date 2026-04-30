"""Q&A interface: load cached index → ask questions."""

import sys

from simple_rag import SimpleRAG, load_dotenv

# ── Config ─────────────────────────────────────────────────────────

_env = load_dotenv()
API_KEY = _env.get("DEEPSEEK_API_KEY", "")
CACHE_FILE = "index.pkl"
VERBOSE = True

# ── Q&A ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not API_KEY:
        print("Missing DEEPSEEK_API_KEY in .env file.")
        sys.exit(1)

    rag = SimpleRAG(api_key=API_KEY, verbose=VERBOSE)

    if not rag.load_cache(CACHE_FILE):
        print("No cache found. Run build.py first.")
        sys.exit(1)

    print(f"=== RAG Q&A (DeepSeek) ===\n"
          f"{len(rag._file_names)} files, {len(rag._chunks)} chunks\n")

    while True:
        try:
            q = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if q.lower() in ("exit", "quit", "q"):
            break
        if not q:
            continue
        answer = rag.ask(q)
        print(f"\n{answer}\n")
