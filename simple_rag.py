"""
Simple RAG (Retrieval-Augmented Generation) framework.
Reads .md files, retrieves relevant chunks, and answers questions using DeepSeek API.
"""

import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

from openai import OpenAI

# ── TF-IDF Retriever ──────────────────────────────────────────────

class TfidfRetriever:
    """Minimal TF-IDF retriever — no external ML dependencies."""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self._vocab: dict[str, int] = {}       # word → id
        self._idf: dict[str, float] = {}       # word → idf
        self._doc_vecs: list[dict[int, float]] = []  # per-chunk sparse tfidf
        self._build()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _build(self):
        tokenized = [self._tokenize(c) for c in self.chunks]
        # vocabulary
        for tokens in tokenized:
            for t in tokens:
                if t not in self._vocab:
                    self._vocab[t] = len(self._vocab)
        N = len(self.chunks)
        # idf
        for word, wid in self._vocab.items():
            df = sum(1 for tokens in tokenized if word in tokens)
            self._idf[word] = math.log((N - df + 0.5) / (df + 0.5) + 1)
        # tfidf vectors
        for tokens in tokenized:
            tf = Counter(tokens)
            total = len(tokens) or 1
            vec: dict[int, float] = {}
            for word, count in tf.items():
                wid = self._vocab.get(word)
                if wid is not None:
                    vec[wid] = (count / total) * self._idf[word]
            self._doc_vecs.append(vec)

    def _query_vec(self, query: str) -> dict[int, float]:
        tokens = self._tokenize(query)
        tf = Counter(tokens)
        total = len(tokens) or 1
        vec: dict[int, float] = {}
        for word, count in tf.items():
            wid = self._vocab.get(word)
            if wid is not None:
                vec[wid] = (count / total) * self._idf.get(word, 0)
        return vec

    def _dot(self, a: dict[int, float], b: dict[int, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(a[wid] * b.get(wid, 0) for wid in a)

    def search(self, query: str, top_k: int = 3) -> list[tuple[str, float]]:
        qv = self._query_vec(query)
        scored = [(c, self._dot(qv, dv)) for c, dv in zip(self.chunks, self._doc_vecs)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def search_diverse(self, query: str, sources: list[str], top_k: int = 10,
                       per_file: int = 2) -> list[tuple[str, float]]:
        """Retrieve top chunks while ensuring each file gets at least 1 slot,
        then up to per_file, then fill with best remaining."""
        qv = self._query_vec(query)
        scored = [(c, self._dot(qv, dv), src)
                  for (c, dv), src in zip(zip(self.chunks, self._doc_vecs), sources)]
        scored.sort(key=lambda x: x[1], reverse=True)

        unique_files = set(sources)
        if len(unique_files) <= 1:
            return [(c, s) for c, s, _ in scored[:top_k]]

        result: list[tuple[str, float]] = []
        file_counts: dict[str, int] = {f: 0 for f in unique_files}

        # Round 1: ensure every file has 1 chunk
        for chunk, score, src in scored:
            if file_counts[src] == 0:
                result.append((chunk, score))
                file_counts[src] = 1
                if len(result) == len(unique_files):
                    break

        # Round 2: add extra chunks up to per_file per file
        for chunk, score, src in scored:
            if len(result) >= top_k:
                break
            if file_counts[src] < per_file:
                result.append((chunk, score))
                file_counts[src] += 1

        # Round 3: fill remaining slots
        for chunk, score, src in scored:
            if len(result) >= top_k:
                break
            if (chunk, score) not in result:
                result.append((chunk, score))
        return result


# ── RAG Engine ────────────────────────────────────────────────────

class SimpleRAG:
    """
    Simple RAG: load .md files → chunk → retrieve → answer with DeepSeek.

    Usage:
        rag = SimpleRAG(api_key="...")
        rag.load("docs/README.md")
        print(rag.ask("What is this project about?"))
    """

    DEEPSEEK_BASE = "https://api.deepseek.com"

    def __init__(self, api_key: str | None = None, model: str = "deepseek-chat",
                 verbose: bool = False):
        key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        self._client = OpenAI(api_key=key, base_url=self.DEEPSEEK_BASE)
        self._verbose = verbose
        self._model = model
        self._chunks: list[str] = []
        self._sources: list[str] = []  # file name per chunk
        self._retriever: TfidfRetriever | None = None
        self._file_names: list[str] = []

    def load(self, *paths: str, chunk_size: int = 500, overlap: int = 100):
        """Load .md files from paths (files or directories)."""
        files: list[Path] = []
        for path in paths:
            p = Path(path)
            if p.is_dir():
                files.extend(sorted(p.rglob("*.md")))
            elif p.suffix == ".md":
                files.append(p)
            else:
                raise ValueError(f"Expected .md file or directory, got: {path}")

        for fp in files:
            text = fp.read_text(encoding="utf-8")
            self._file_names.append(fp.name)
            new_chunks = self._chunk(text, chunk_size, overlap)
            self._chunks.extend(new_chunks)
            self._sources.extend([fp.name] * len(new_chunks))
        self._retriever = TfidfRetriever(self._chunks)

    def _chunk(self, text: str, chunk_size: int, overlap: int) -> list[str]:
        """Split text into overlapping chunks, respecting paragraph boundaries."""
        paragraphs = re.split(r"\n\n+", text.strip())
        chunks: list[str] = []
        current = ""
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if len(current) + len(para) <= chunk_size:
                current = (current + "\n\n" + para).strip()
            else:
                if current:
                    chunks.append(current)
                # sliding window with overlap
                step = chunk_size - overlap
                for i in range(0, len(para), step):
                    chunks.append(para[i : i + chunk_size])
                current = ""
        if current:
            chunks.append(current)
        return [c for c in chunks if len(c) > 20]

    def ask(self, question: str, top_k: int = 14) -> str:
        """Ask a question about the loaded documents."""
        if not self._retriever or not self._chunks:
            return "No documents loaded. Call .load() first."

        results = self._retriever.search_diverse(question, self._sources, top_k=top_k, per_file=2)
        context = "\n\n---\n\n".join(chunk for chunk, _ in results)

        if self._verbose:
            print(f"[DEBUG] Query: {question}")
            for i, (chunk, score) in enumerate(results):
                # use index in _chunks to find source
                try:
                    idx = self._chunks.index(chunk)
                    src = self._sources[idx]
                except ValueError:
                    src = "?"
                print(f"[DEBUG]   [{i}] score={score:.4f}  file={src}  text={chunk[:80]}...")

        prompt = (
            "You are a helpful assistant. Answer the user's question based ONLY on "
            "the provided document excerpts. If the documents don't contain enough "
            "information, say so honestly.\n\n"
            f"### Documents\n{context}\n\n"
            f"### Question\n{question}\n\n"
            "### Answer"
        )

        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.choices[0].message.content


# ── Helpers ────────────────────────────────────────────────────────

def _load_dotenv(path: str = ".env") -> dict[str, str]:
    """Load KEY=VALUE pairs from a .env file."""
    env: dict[str, str] = {}
    p = Path(path)
    if p.is_file():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


# ── Config ─────────────────────────────────────────────────────────

_env = _load_dotenv()
API_KEY = _env.get("DEEPSEEK_API_KEY", "")
MD_DIR = "knowledge/"
VERBOSE = True

# ── CLI ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if not API_KEY:
        print("Missing DEEPSEEK_API_KEY in .env file.")
        sys.exit(1)

    rag = SimpleRAG(api_key=API_KEY, verbose=VERBOSE)

    print(f"=== Simple RAG (DeepSeek) ===\n")
    rag.load(MD_DIR)
    print(f"Loaded {len(rag._file_names)} file(s), {len(rag._chunks)} chunks\n")

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
