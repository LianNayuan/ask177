"""Q&A interface: load cached index → ask questions.

Usage:
  python ask.py              # verbose mode (title/rewrite logs, no chunk dump)
  python ask.py --debug       # verbose + per-chunk retrieval dump
  python ask.py -q            # quiet mode (only answers, no logs)
"""

import sys
from pathlib import Path

from simple_rag import SimpleRAG, load_dotenv

# ── Config ─────────────────────────────────────────────────────────

_env = load_dotenv()
API_KEY = _env.get("DEEPSEEK_API_KEY", "")
CACHE_FILE = "index.pkl"
GLOSSARY_FILE = "knowledge/glossary.md"


def _save_glossary(glossary: dict[str, str], path: str):
    """Persist glossary to file."""
    lines = ["# 口语化词汇对照表"]
    for slang, formal in glossary.items():
        lines.append(f"{slang} | {formal}")
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


# ── Q&A ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not API_KEY:
        print("Missing DEEPSEEK_API_KEY in .env file.")
        sys.exit(1)

    # Parse flags
    verbose = True
    debug_chunks = False
    for a in sys.argv[1:]:
        if a == "--debug":
            debug_chunks = True
        elif a == "-q":
            verbose = False
        else:
            print(f"Unknown argument: {a}")
            sys.exit(1)

    rag = SimpleRAG(api_key=API_KEY, verbose=verbose)

    if not rag.load_cache(CACHE_FILE):
        print("No cache found. Run build_tfidf.py first.")
        sys.exit(1)

    # Show retrieval mode
    if rag._dense_retriever is not None:
        mode = f"TF-IDF + Dense ({rag._embedding_model_used or 'unknown'}, weight={rag._dense_weight})"
    else:
        mode = "TF-IDF only (no dense embeddings)"
        print("  Hint: run 'python build_embeddings.py' for semantic search.")

    print(f"\n=== RAG Q&A (DeepSeek) ===\n"
          f"{len(rag._file_names)} files, {len(rag._chunks)} chunks\n"
          f"Retrieval: {mode}\n"
          f"Commands: /add slang=formal  /list  /del slang\n")

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

        # ── Commands ──────────────────────────────────────────────
        if q.startswith("/add "):
            parts = q[5:].split("=", 1)
            if len(parts) == 2:
                slang, formal = parts[0].strip(), parts[1].strip()
                rag._glossary[slang] = formal
                _save_glossary(rag._glossary, GLOSSARY_FILE)
                print(f"  + {slang} → {formal}\n")
            else:
                print("  Usage: /add 大招=特殊武器\n")
            continue

        if q == "/list":
            if rag._glossary:
                for s, f in rag._glossary.items():
                    print(f"  {s} → {f}")
            else:
                print("  (empty)")
            print()
            continue

        if q.startswith("/del "):
            slang = q[5:].strip()
            if slang in rag._glossary:
                del rag._glossary[slang]
                _save_glossary(rag._glossary, GLOSSARY_FILE)
                print(f"  - {slang} removed\n")
            else:
                print(f"  '{slang}' not found\n")
            continue

        # ── Ask ───────────────────────────────────────────────────
        answer = rag.ask(q, debug_chunks=debug_chunks)
        print(f"\n{answer}\n")
