"""Build TF-IDF index: load files → chunk → cache to disk. Run once, or when files change.

Usage:
  python build_tfidf.py              # auto-detect changed files, incremental update
  python build_tfidf.py --file X.md  # force re-process specific file(s)
  python build_tfidf.py -f X.md Y.md # same, multiple files

For dense embeddings, run python build_embeddings.py separately.
"""

import sys
from pathlib import Path

from simple_rag import SimpleRAG, load_dotenv

# ── Config ─────────────────────────────────────────────────────────

_env = load_dotenv()
API_KEY = _env.get("DEEPSEEK_API_KEY", "")
MD_DIRS = ["knowledge/wiki_cn", "knowledge/picture_ocr"]
GLOSSARY_FILE = "knowledge/glossary.md"
CACHE_FILE = "index.pkl"

# ── Build ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not API_KEY:
        print("Missing DEEPSEEK_API_KEY in .env file.")
        sys.exit(1)

    # Parse args
    force_files: list[str] = []
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] in ("--file", "-f") and i + 1 < len(args):
            force_files.append(args[i + 1])
            i += 2
        else:
            print(f"Unknown argument: {args[i]}")
            sys.exit(1)

    rag = SimpleRAG(api_key=API_KEY, verbose=True)

    # ── Incremental update (cache exists) ──────────────────────────
    if Path(CACHE_FILE).exists():
        rag.load_cache(CACHE_FILE, force=True)

        # Mark forced files as "modified" so incremental_update reprocesses them
        for fname in force_files:
            matched: list[str] = []
            for fp in rag._file_mtimes:
                fp_name = Path(fp).name
                fp_stem = Path(fp).stem
                if fname in (fp, fp_name, fp_stem, fp_name.replace(".md", ""),
                             f"{fname}.md", f"knowledge/wiki_cn/{fname}"):
                    matched.append(fp)
            if not matched:
                print(f"Warning: '{fname}' not found in cache, skipping.")
            for fp in matched:
                rag._file_mtimes[fp] = 0  # force reprocess

        # Reload glossary if it changed since the cache was built
        glossary_reloaded = False
        glossary_path = Path(GLOSSARY_FILE)
        cache_mtime = Path(CACHE_FILE).stat().st_mtime
        if glossary_path.exists() and glossary_path.stat().st_mtime > cache_mtime + 0.1:
            rag._glossary = {}
            rag.load_glossary(GLOSSARY_FILE)
            glossary_reloaded = True

        updated = rag.incremental_update(*MD_DIRS)

        if not updated and not glossary_reloaded:
            print(f"Cache is up-to-date ({CACHE_FILE}). Nothing to build.")
            print("Use --file <name> to force re-process a specific file,"
                  " or delete index.pkl for a full rebuild.")
            sys.exit(0)

        rag.save_cache(CACHE_FILE)
        print("Done (incremental). Run ask.py to start Q&A.")
        sys.exit(0)

    # ── Full build (no cache exists) ───────────────────────────────
    if force_files:
        print("Warning: --file has no effect on first build (no cache exists).")

    print(f"Building index from {MD_DIRS} ...")
    rag.load(*MD_DIRS)
    rag.load_glossary(GLOSSARY_FILE)
    print(f"  {len(rag._file_names)} files → {len(rag._chunks)} chunks"
          f"  glossary: {len(rag._glossary)} entries")

    rag.save_cache(CACHE_FILE)
    print("Done. Run ask.py to start Q&A.")
