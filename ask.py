"""Q&A interface: load cached index → ask questions.

Usage:
  python ask.py              # verbose mode (title/rewrite logs, no chunk dump)
  python ask.py --debug       # verbose + per-chunk retrieval dump
  python ask.py -q            # quiet mode (only answers, no logs)
"""

import sys
import time
from pathlib import Path

from database import Database
from simple_rag import SimpleRAG, load_dotenv

# ── Config ─────────────────────────────────────────────────────────

_env = load_dotenv()
API_KEY = _env.get("DEEPSEEK_API_KEY", "")
CACHE_FILE = "index.pkl"
GLOSSARY_FILE = "knowledge/glossary.md"


def _save_glossary_file(glossary: dict[str, str], path: str):
    """Sync glossary to file for backward compatibility."""
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

    # Init database
    db = Database()
    stats = db.stats()
    print(f"[DB] Connected to data.db ({stats['total_queries']} queries,"
          f" {stats['glossary_entries']} glossary entries)")

    # Import glossary from file on first run, then use DB as source of truth
    if stats["glossary_entries"] == 0:
        n = db.import_glossary_from_file(GLOSSARY_FILE)
        if n > 0:
            print(f"[DB] Imported {n} glossary entries from {GLOSSARY_FILE}")

    rag = SimpleRAG(api_key=API_KEY, verbose=verbose)

    if not rag.load_cache(CACHE_FILE, db=db):
        print("No cache found. Run build_tfidf.py first.")
        sys.exit(1)

    # Load glossary from DB into RAG
    rag._glossary = db.load_glossary()

    # Show retrieval mode
    if rag._chroma_retriever is not None:
        mode = f"TF-IDF + Dense/ChromaDB ({rag._embedding_model_used or 'unknown'}, weight={rag._dense_weight})"
    elif rag._dense_retriever is not None:
        mode = f"TF-IDF + Dense ({rag._embedding_model_used or 'unknown'}, weight={rag._dense_weight})"
    else:
        mode = "TF-IDF only (no dense embeddings)"
        print("  Hint: run 'python build_embeddings.py --chroma' for semantic search.")

    print(f"\n=== RAG Q&A (DeepSeek) ===\n"
          f"{len(rag._file_names)} files, {len(rag._chunks)} chunks\n"
          f"Retrieval: {mode}\n"
          f"Commands: /add  /list  /del  /history  /stats  /feedback\n")

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
                db.add_glossary(slang, formal)
                _save_glossary_file(rag._glossary, GLOSSARY_FILE)
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
                db.del_glossary(slang)
                _save_glossary_file(rag._glossary, GLOSSARY_FILE)
                print(f"  - {slang} removed\n")
            else:
                print(f"  '{slang}' not found\n")
            continue

        if q == "/history":
            rows = db.recent_queries(20)
            if not rows:
                print("  (no queries yet)\n")
            else:
                for r in rows:
                    q_preview = r["question"][:60]
                    print(f"  [{r['id']}] {r['created_at']} | {r['mode']} | {r['latency_ms']}ms")
                    print(f"       {q_preview}...")
                print()
            continue

        if q == "/stats":
            s = db.stats()
            print(f"  Total queries:    {s['total_queries']}")
            print(f"  Avg latency:      {s['avg_latency_ms']}ms")
            print(f"  Glossary entries: {s['glossary_entries']}")
            print(f"  Retrieval modes:  {s['retrieval_modes']}")
            print()
            continue

        if q.startswith("/feedback "):
            parts = q[10:].split()
            if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
                qid, rating = int(parts[0]), int(parts[1])
                comment = " ".join(parts[2:]) if len(parts) > 2 else ""
                if 1 <= rating <= 5:
                    db.add_feedback(qid, rating, comment)
                    print(f"  Feedback recorded: query #{qid} → {rating}/5\n")
                else:
                    print("  Rating must be 1-5\n")
            else:
                print("  Usage: /feedback <query_id> <1-5> [comment]\n")
            continue

        # ── Ask ───────────────────────────────────────────────────
        t0 = time.time()
        answer = rag.ask(q, debug_chunks=debug_chunks)
        elapsed_ms = int((time.time() - t0) * 1000)

        # Log to SQLite (sanitize in case API returned surrogates)
        info = rag._last_query_info
        db.log_query(
            question=SimpleRAG._sanitize(q),
            answer=SimpleRAG._sanitize(answer),
            mode=info.get("mode", ""),
            hit_files=info.get("hit_files", ""),
            rewrite=info.get("rewrite", ""),
            latency_ms=elapsed_ms,
        )

        print(f"\n{answer}\n")
        if verbose:
            print(f"  [{elapsed_ms}ms, {info.get('mode', '?')},"
                  f" hits: {info.get('hit_files', '?')}]")
            if info.get("rewrite"):
                print(f"  [rewrite: {info['question']} → {info['rewrite']}]")
            print()
