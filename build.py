"""Build index: load files → chunk → cache to disk. Run once, or when files change."""

import sys
from pathlib import Path

from simple_rag import SimpleRAG, load_dotenv

# ── Config ─────────────────────────────────────────────────────────

_env = load_dotenv()
API_KEY = _env.get("DEEPSEEK_API_KEY", "")
MD_DIR = "knowledge/"
CACHE_FILE = "index.pkl"

# ── Build ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not API_KEY:
        print("Missing DEEPSEEK_API_KEY in .env file.")
        sys.exit(1)

    rag = SimpleRAG(api_key=API_KEY, verbose=True)

    # Check if cache is fresh
    if rag.is_fresh(CACHE_FILE):
        print(f"Cache is up-to-date ({CACHE_FILE}). Nothing to build.")
        print("Delete index.pkl or modify source files to force rebuild.")
        sys.exit(0)

    print(f"Building index from {MD_DIR!r} ...")
    rag.load(MD_DIR)
    print(f"  {len(rag._file_names)} files → {len(rag._chunks)} chunks")

    rag.save_cache(CACHE_FILE)
    print("Done. Run ask.py to start Q&A.")
