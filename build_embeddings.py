"""Pre-compute dense embeddings for hybrid retrieval.

Run AFTER python build_tfidf.py (which creates the TF-IDF index).
Supports three storage backends: in-memory (pkl), ChromaDB, or API.

Usage:
  python build_embeddings.py                     # in-memory, BAAI/bge-small-zh-v1.5
  python build_embeddings.py --chroma             # ChromaDB storage (recommended)
  python build_embeddings.py --mode api           # DeepSeek API embeddings
  python build_embeddings.py --model <name>       # override model
  python build_embeddings.py --force              # overwrite existing embeddings

Examples:
  python build_embeddings.py --chroma                         # ChromaDB + local model
  python build_embeddings.py --chroma --mode api              # ChromaDB + API
  python build_embeddings.py --chroma --model ./my-finetuned  # ChromaDB + fine-tuned model
"""

import sys
from pathlib import Path

from database import Database
from simple_rag import ChromaRetriever, DenseRetriever, SimpleRAG, load_dotenv

# ── Defaults ─────────────────────────────────────────────────────────

DEFAULT_LOCAL_MODEL = "BAAI/bge-small-zh-v1.5"
DEFAULT_API_MODEL = "deepseek-embedding-lite"
CACHE_FILE = "index.pkl"
CHROMA_DIR = "chroma_db"

# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _env = load_dotenv()
    api_key = _env.get("DEEPSEEK_API_KEY", "")

    # Parse args
    mode = "local"
    model_override: str | None = None
    force = False
    use_chroma = False
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
        elif args[i] == "--chroma":
            use_chroma = True
            i += 1
        elif args[i] == "--force":
            force = True
            i += 1
        else:
            print(f"Unknown argument: {args[i]}")
            print("Usage: python build_embeddings.py [--chroma] [--mode local|api]"
                  " [--model NAME] [--force]")
            sys.exit(1)

    # Check prerequisites
    if not Path(CACHE_FILE).exists():
        print(f"Error: {CACHE_FILE} not found. Run 'python build_tfidf.py' first.")
        sys.exit(1)

    # Load existing index
    db = Database()
    rag = SimpleRAG(api_key=api_key, verbose=True)
    rag.load_cache(CACHE_FILE, force=True, db=db)

    print(f"Loaded {len(rag._chunks)} chunks from {CACHE_FILE}")

    # Check if embeddings already exist
    existing = False
    if use_chroma:
        chroma_path = Path(CHROMA_DIR)
        if chroma_path.exists() and any(chroma_path.iterdir()):
            existing = True
            existing_desc = f"ChromaDB at {CHROMA_DIR}/"
    else:
        if rag._dense_retriever is not None:
            existing = True
            existing_desc = f"{len(rag._embeddings)} in-memory vectors"

    if existing and not force:
        print(f"Embeddings already exist ({existing_desc}). Use --force to overwrite.")
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
    rag._embedding_model_used = model_name

    if use_chroma:
        # Delete old collection if it exists (for --force)
        if force and Path(CHROMA_DIR).exists():
            import shutil
            shutil.rmtree(CHROMA_DIR)
            print(f"  Removed old ChromaDB at {CHROMA_DIR}/")

        rag._chroma_retriever = ChromaRetriever(
            chunks=rag._chunks,
            embeddings=embeddings_list,
            sources=rag._sources,
            persist_dir=CHROMA_DIR,
        )
        rag._chroma_db = CHROMA_DIR
        rag._embeddings = []  # don't duplicate in pkl
        rag._dense_retriever = None
        rag.save_cache(CACHE_FILE, db=db)
        print(f"\n✓ {len(rag._chunks)} chunks indexed in ChromaDB"
              f" ({len(embeddings_list[0])}-dim) via {model_name}")
        print(f"  ChromaDB: {CHROMA_DIR}/  |  index: {CACHE_FILE} (TF-IDF only)")
    else:
        rag._embeddings = embeddings_list
        rag._dense_retriever = DenseRetriever(rag._chunks, rag._embeddings)
        rag._chroma_db = None
        rag._chroma_retriever = None
        rag.save_cache(CACHE_FILE, db=db)
        print(f"\n✓ {len(rag._embeddings)} dense vectors"
              f" ({len(rag._embeddings[0])}-dim) via {model_name}")
        print(f"  Saved to {CACHE_FILE}")

    print(f"  Run ask.py — you should see 'TF-IDF + Dense' in the banner.")
