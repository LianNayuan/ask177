"""Q&A interface: load cached index → ask questions."""

import sys
from pathlib import Path

from simple_rag import SimpleRAG, load_dotenv

# ── Config ─────────────────────────────────────────────────────────

_env = load_dotenv()
API_KEY = _env.get("DEEPSEEK_API_KEY", "")
CACHE_FILE = "index.pkl"
GLOSSARY_FILE = "knowledge/glossary.md"
VERBOSE = True


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

    rag = SimpleRAG(api_key=API_KEY, verbose=VERBOSE)

    if not rag.load_cache(CACHE_FILE):
        print("No cache found. Run build.py first.")
        sys.exit(1)

    print(f"=== RAG Q&A (DeepSeek) ===\n"
          f"{len(rag._file_names)} files, {len(rag._chunks)} chunks\n"
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
        answer = rag.ask(q)
        print(f"\n{answer}\n")
