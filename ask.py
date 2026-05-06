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
    # Windows: force UTF-8 to avoid GBK mojibake
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stdin.reconfigure(encoding="utf-8")
        except Exception:
            pass

    if not API_KEY:
        print("Missing DEEPSEEK_API_KEY in .env file.")
        sys.exit(1)

    # Parse flags
    verbose = True
    debug_chunks = False
    retrieval_mode = "hybrid"
    dense_weight = 0.5
    agentic = False
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--debug":
            debug_chunks = True
            i += 1
        elif a == "-q":
            verbose = False
            i += 1
        elif a == "--agentic":
            agentic = True
            i += 1
        elif a == "--mode" and i + 1 < len(args):
            retrieval_mode = args[i + 1]
            if retrieval_mode not in ("tfidf", "dense", "hybrid"):
                print(f"Invalid --mode: {retrieval_mode!r}. Use: tfidf, dense, hybrid")
                sys.exit(1)
            i += 2
        elif a == "--dense-weight" and i + 1 < len(args):
            dense_weight = float(args[i + 1])
            if not (0.0 <= dense_weight <= 1.0):
                print(f"Invalid --dense-weight: {dense_weight}. Must be 0.0 - 1.0")
                sys.exit(1)
            i += 2
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

    rag = SimpleRAG(api_key=API_KEY, verbose=verbose,
                    retrieval_mode=retrieval_mode,
                    dense_weight=dense_weight,
                    agentic=agentic)

    if not rag.load_cache(CACHE_FILE, db=db):
        print("No cache found. Run build_tfidf.py first.")
        sys.exit(1)

    # Load glossary from DB into RAG
    rag._glossary = db.load_glossary()

    # Show retrieval mode
    if rag._retrieval_mode == "tfidf":
        mode = "TF-IDF only (forced by --mode)"
    elif rag._retrieval_mode == "dense":
        if rag._chroma_retriever is not None:
            mode = f"Dense only / ChromaDB ({rag._embedding_model_used or 'unknown'})"
        elif rag._dense_retriever is not None:
            mode = f"Dense only / in-memory ({rag._embedding_model_used or 'unknown'})"
        else:
            mode = "Dense only (NO dense index available!)"
    elif rag._chroma_retriever is not None:
        mode = f"TF-IDF + Dense/ChromaDB ({rag._embedding_model_used or 'unknown'}, weight={rag._dense_weight})"
    elif rag._dense_retriever is not None:
        mode = f"TF-IDF + Dense/in-memory ({rag._embedding_model_used or 'unknown'}, weight={rag._dense_weight})"
    else:
        mode = "TF-IDF only (no dense embeddings)"
        print("  Hint: run 'python build_embeddings.py --chroma' for semantic search.")

    # ── Session management ──────────────────────────────────────────
    conv_id = db.last_conversation_id()
    resume_info = ""
    if conv_id:
        conv = db.get_conversation(conv_id)
        if conv:
            title_preview = (conv["title"] or "(empty)")[:30]
            resume_info = f" (resumed #{conv_id}: {title_preview})"
    else:
        conv_id = db.create_conversation()
        resume_info = " (new session)"

    def _history_messages(n: int = 10) -> list[dict[str, str]]:
        rows = db.get_history(conv_id, limit=n)
        return [{"role": r["role"], "content": r["content"]} for r in rows]

    
    agentic_tag = " [agentic]" if agentic else ""
    print(f"\n=== RAG Q&A (DeepSeek){agentic_tag} ===\n"
          f"{len(rag._file_names)} files, {len(rag._chunks)} chunks\n"
          f"Retrieval: {mode}\n"
          f"Session #{conv_id}{resume_info}\n"
          f"Commands: /new  /sessions  /switch  /add  /list  /del"
          f"  /history  /stats  /feedback\n")

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
        if q == "/new":
            conv_id = db.create_conversation()
            print(f"  New session #{conv_id} started.\n")
            continue

        if q == "/sessions":
            rows = db.recent_conversations(10)
            if not rows:
                print("  (no sessions)\n")
            else:
                for r in rows:
                    marker = " ← current" if r["id"] == conv_id else ""
                    title = (r["title"] or "(empty)")[:40]
                    print(f"  [#{r['id']}] {r['updated_at']}"
                          f" | {r['msg_count']} msgs | {title}{marker}")
                print()
            continue

        if q.startswith("/switch "):
            target = q[8:].strip()
            if target.isdigit():
                target_id = int(target)
                if db.get_conversation(target_id):
                    conv_id = target_id
                    conv = db.get_conversation(conv_id)
                    print(f"  Switched to session #{conv_id}"
                          f" ({(conv['title'] or '')[:30]})\n")
                else:
                    print(f"  Session #{target_id} not found.\n")
            else:
                print("  Usage: /switch <id>\n")
            continue

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
            print(f"  Errors:           {s.get('error_count', 0)}")
            print(f"  Sessions:         {s.get('session_count', 0)}")
            print(f"  Glossary entries: {s['glossary_entries']}")
            pt = s.get('prompt_tokens', 0)
            ct = s.get('completion_tokens', 0)
            print(f"  Prompt tokens:    {pt:,}")
            print(f"  Completion tokens:{ct:,}")
            print(f"  Total tokens:     {s.get('total_tokens', 0):,}")
            # Estimate: deepseek-chat ~$0.14/1M input, $0.28/1M output
            cost = pt / 1_000_000 * 0.14 + ct / 1_000_000 * 0.28
            print(f"  Est. cost (USD):  ${cost:.4f}")
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
        history = _history_messages(10)
        answer = rag.ask(q, debug_chunks=debug_chunks, history=history)
        elapsed_ms = int((time.time() - t0) * 1000)

        # Save to conversation history
        db.add_message(conv_id, "user", SimpleRAG._sanitize(q))
        db.add_message(conv_id, "assistant", SimpleRAG._sanitize(answer))

        # Log to query_logs
        info = rag._last_query_info
        error = info.get("error", "")
        qid = db.log_query(
            question=SimpleRAG._sanitize(q),
            answer=SimpleRAG._sanitize(answer),
            mode=info.get("mode", ""),
            hit_files=info.get("hit_files", ""),
            rewrite=info.get("rewrite", ""),
            latency_ms=elapsed_ms,
            session_id=conv_id,
            prompt_tokens=info.get("prompt_tokens", 0),
            completion_tokens=info.get("completion_tokens", 0),
            error=error,
        )

        print(f"\n{answer}\n")
        if verbose:
            pt = info.get("prompt_tokens", 0)
            ct = info.get("completion_tokens", 0)
            cost = pt / 1_000_000 * 0.14 + ct / 1_000_000 * 0.28
            print(f"  [{elapsed_ms}ms, {info.get('mode', '?')},"
                  f" hits: {info.get('hit_files', '?')}]")
            print(f"  [tokens: {pt}↑ + {ct}↓ = {pt + ct}, "
                  f"est. ${cost:.5f}]")
            if info.get("rewrite"):
                print(f"  [rewrite: {info['question']} → {info['rewrite']}]")
            print()
