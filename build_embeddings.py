"""Pre-compute dense embeddings and add them to index.pkl.

Run AFTER python build_tfidf.py (which creates the TF-IDF index).
Supports two modes: local model (default) or API.

Usage:
  python build_embeddings.py                   # local mode, BAAI/bge-small-zh-v1.5
  python build_embeddings.py --mode local       # explicit local mode
  python build_embeddings.py --mode api         # DeepSeek embedding API
  python build_embeddings.py --model <name>     # override model
  python build_embeddings.py --force            # overwrite existing embeddings

Examples:
  # Use a fine-tuned local model
  python build_embeddings.py --model ./my-finetuned-bge

  # Use DeepSeek API with a specific model
  python build_embeddings.py --mode api --model deepseek-embedding-lite

Fine-tuning tip:
  You can fine-tune BAAI/bge-small-zh-v1.5 on domain-specific data (e.g. weapon
  synonym pairs, question-chunk pairs) using sentence-transformers or FlagEmbedding.
  Then pass the fine-tuned model path via --model.
"""

import sys
from pathlib import Path

from simple_rag import DenseRetriever, SimpleRAG, load_dotenv

# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"  # 24M params, 512-dim, Chinese-optimized
DEFAULT_API_MODEL = "deepseek-embedding-lite"
CACHE_FILE = "index.pkl"

# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _env = load_dotenv()
    api_key = _env.get("DEEPSEEK_API_KEY", "")

    # Parse args
    mode = "local"
    model_override: str | None = None
    force = False
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--mode" and i + 1 < len(args):
            mode = args[i + 1]
            if mode not in ("local", "api"):
                print(f"Unknown mode: {mode}. Use 'local' or 'api'.")
                sys.exit(1)
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model_override = args[i + 1]
            i += 2
        elif args[i] == "--force":
            force = True
            i += 1
        else:
            print(f"Unknown argument: {args[i]}")
            print("Usage: python build_embeddings.py [--mode local|api] [--model NAME] [--force]")
            sys.exit(1)

    # Check prerequisites
    if not Path(CACHE_FILE).exists():
        print(f"Error: {CACHE_FILE} not found. Run 'python build_tfidf.py' first.")
        sys.exit(1)

    # Load existing index
    rag = SimpleRAG(api_key=api_key, verbose=True)
    rag.load_cache(CACHE_FILE, force=True)

    print(f"Loaded {len(rag._chunks)} chunks from {CACHE_FILE}")

    # Check if embeddings already exist
    if rag._dense_retriever is not None and not force:
        print(f"Embeddings already exist ({len(rag._embeddings)} vectors,"
              f" {len(rag._embeddings[0])}-dim).")
        print("Use --force to overwrite.")
        sys.exit(0)

    # Resolve model name
    if model_override:
        model_name = model_override
    elif mode == "local":
        model_name = DEFAULT_LOCAL_MODEL
    else:
        model_name = DEFAULT_API_MODEL

    # ── Generate embeddings ──────────────────────────────────────────
    if mode == "local":
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            print("sentence-transformers not installed. Run: pip install sentence-transformers")
            print("Or use --mode api for DeepSeek API embeddings.")
            sys.exit(1)

        print(f"Loading local model: {model_name}")
        model = SentenceTransformer(model_name)

        print(f"Embedding {len(rag._chunks)} chunks (this may take a few minutes)...")
        embeddings = model.encode(
            rag._chunks,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        embeddings_list = embeddings.tolist()

    else:  # mode == "api"
        if not api_key:
            print("Missing DEEPSEEK_API_KEY in .env file.")
            sys.exit(1)

        print(f"Embedding {len(rag._chunks)} chunks via DeepSeek API"
              f" (model: {model_name})...")
        rag._embedding_model = model_name
        embeddings_list = rag._embed_chunks(rag._chunks)

    # ── Save ─────────────────────────────────────────────────────────
    rag._embeddings = embeddings_list
    rag._embedding_model_used = model_name
    rag._dense_retriever = DenseRetriever(rag._chunks, rag._embeddings)

    rag.save_cache(CACHE_FILE)
    print(f"\n✓ {len(rag._embeddings)} dense vectors ({len(rag._embeddings[0])}-dim)"
          f" via {model_name}")
    print(f"  Saved to {CACHE_FILE}")
    print(f"  Run ask.py — you should see 'TF-IDF + Dense' in the banner.")
