"""
Simple RAG (Retrieval-Augmented Generation) framework.
Core library: TfidfRetriever + SimpleRAG with pickle-based caching.
"""

import math
import os
import pickle
import re
import time
from collections import Counter
from pathlib import Path

from openai import APIConnectionError, OpenAI


# ── TF-IDF Retriever ──────────────────────────────────────────────

class TfidfRetriever:
    """Minimal TF-IDF retriever — no external ML dependencies."""

    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self._vocab: dict[str, int] = {}
        self._idf: dict[str, float] = {}
        self._doc_vecs: list[dict[int, float]] = []
        self._build()

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def _build(self):
        tokenized = [self._tokenize(c) for c in self.chunks]
        for tokens in tokenized:
            for t in tokens:
                if t not in self._vocab:
                    self._vocab[t] = len(self._vocab)
        N = len(self.chunks)
        for word, wid in self._vocab.items():
            df = sum(1 for tokens in tokenized if word in tokens)
            self._idf[word] = math.log((N - df + 0.5) / (df + 0.5) + 1)
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
                       per_file: int = 2,
                       file_filter: set[str] | None = None) -> list[tuple[str, float]]:
        """Retrieve top chunks while ensuring each source file gets at least 1 slot.
        If file_filter is given, only consider chunks from those files."""
        qv = self._query_vec(query)
        scored = [(c, self._dot(qv, dv), src)
                  for (c, dv), src in zip(zip(self.chunks, self._doc_vecs), sources)]
        if file_filter:
            scored = [(c, s, src) for c, s, src in scored if src in file_filter]
            if not scored:
                return []
        scored.sort(key=lambda x: x[1], reverse=True)

        unique_files = {src for _, _, src in scored}
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

    # ── pickle helpers ───────────────────────────────────────────

    def _state(self) -> dict:
        return {"vocab": self._vocab, "idf": self._idf,
                "doc_vecs": self._doc_vecs, "chunks": self.chunks}

    @classmethod
    def _from_state(cls, state: dict) -> "TfidfRetriever":
        obj = cls.__new__(cls)
        obj._vocab = state["vocab"]
        obj._idf = state["idf"]
        obj._doc_vecs = state["doc_vecs"]
        obj.chunks = state["chunks"]
        return obj


# ── RAG Engine ────────────────────────────────────────────────────

class SimpleRAG:
    """
    Simple RAG: load .md files → chunk → retrieve → answer with DeepSeek.

    Usage:
        rag = SimpleRAG()
        rag.load("knowledge/")
        rag.save_cache("index.pkl")

        # later:
        rag = SimpleRAG()
        rag.load_cache("index.pkl")
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
        self._sources: list[str] = []
        self._file_names: list[str] = []
        self._file_mtimes: dict[str, float] = {}
        self._titles: dict[str, str] = {}           # file name → title
        self._nicknames: dict[str, list[str]] = {}   # file name → nicknames
        self._glossary: dict[str, str] = {}          # slang → formal
        self._retriever: TfidfRetriever | None = None

    # ── Build ────────────────────────────────────────────────────

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
            self._file_mtimes[str(fp)] = fp.stat().st_mtime
            text = fp.read_text(encoding="utf-8")
            self._file_names.append(fp.name)
            title = self._extract_title(text) or fp.stem
            self._titles[fp.name] = title
            nicks = self._extract_nicknames(text)
            if nicks:
                self._nicknames[fp.name] = nicks
            new_chunks = self._chunk(text, chunk_size, overlap)
            # Prepend title to each chunk so retriever + LLM know the source
            prefixed = [f"[{title}]\n{c}" for c in new_chunks]
            self._chunks.extend(prefixed)
            self._sources.extend([fp.name] * len(prefixed))
        self._retriever = TfidfRetriever(self._chunks)

    def load_glossary(self, path: str = "knowledge/glossary.md"):
        """Load slang→formal mappings from a glossary file.
        Format per line: 口语词 | 正式词"""
        p = Path(path)
        if not p.is_file():
            return
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                slang, formal = line.split("|", 1)
                self._glossary[slang.strip()] = formal.strip()
        if self._verbose and self._glossary:
            print(f"[Glossary] Loaded {len(self._glossary)} mappings")

    def _rewrite_query(self, question: str) -> str:
        """Replace slang terms in the question with formal equivalents."""
        result = question
        for slang, formal in self._glossary.items():
            if slang in result:
                result = result.replace(slang, formal)
                if self._verbose:
                    print(f"[Rewrite] '{slang}' → '{formal}'")
        return result

    @staticmethod
    def _extract_title(text: str) -> str | None:
        """Extract the first # heading, trimmed to the part before ' - '."""
        m = re.search(r"^#\s+(.+)", text, re.MULTILINE)
        if not m:
            return None
        title = m.group(1).strip()
        title = re.sub(r"\s*[-—–]\s*(完整)?武器详情.*$", "", title)
        return title

    @staticmethod
    def _extract_nicknames(text: str) -> list[str]:
        """Extract nicknames (俗称) from the markdown, e.g. '俗称：红牙刷'."""
        m = re.search(r"俗称[：:]\s*(.+)", text)
        if not m:
            return []
        raw = m.group(1).strip()
        if raw in ("—", "—", "-", "无", ""):
            return []
        # Split on Chinese/English commas and enumeration markers
        parts = re.split(r"[、，,，]", raw)
        return [p.strip() for p in parts if p.strip()]

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
                step = chunk_size - overlap
                for i in range(0, len(para), step):
                    chunks.append(para[i : i + chunk_size])
                current = ""
        if current:
            chunks.append(current)
        return [c for c in chunks if len(c) > 20]

    # ── Cache ────────────────────────────────────────────────────

    def save_cache(self, path: str = "index.pkl"):
        """Save chunks and retriever state to a pickle file."""
        if not self._retriever:
            raise RuntimeError("No index built. Call .load() first.")
        data = {
            "chunks": self._chunks,
            "sources": self._sources,
            "file_names": self._file_names,
            "file_mtimes": self._file_mtimes,
            "titles": self._titles,
            "nicknames": self._nicknames,
            "glossary": self._glossary,
            "retriever": self._retriever._state(),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        if self._verbose:
            print(f"[Cache] Saved {len(self._chunks)} chunks to {path}")

    def load_cache(self, path: str = "index.pkl") -> bool:
        """Load cached chunks and retriever. Returns True if cache is fresh."""
        if not Path(path).exists():
            if self._verbose:
                print(f"[Cache] No cache file at {path}")
            return False

        with open(path, "rb") as f:
            data = pickle.load(f)

        # Check freshness: all source files unchanged
        stale = []
        for fpath, mtime in data.get("file_mtimes", {}).items():
            if not Path(fpath).exists() or Path(fpath).stat().st_mtime > mtime + 0.1:
                stale.append(fpath)
        if stale:
            if self._verbose:
                print(f"[Cache] Stale — {len(stale)} file(s) changed: {stale[:3]}...")
            return False

        self._chunks = data["chunks"]
        self._sources = data["sources"]
        self._file_names = data["file_names"]
        self._file_mtimes = data["file_mtimes"]
        self._titles = data.get("titles", {})
        self._nicknames = data.get("nicknames", {})
        self._glossary = data.get("glossary", {})
        self._retriever = TfidfRetriever._from_state(data["retriever"])
        if self._verbose:
            print(f"[Cache] Loaded {len(self._chunks)} chunks from {path}")
        return True

    def is_fresh(self, cache_path: str = "index.pkl") -> bool:
        """Check if cache exists and is up-to-date without loading."""
        if not Path(cache_path).exists():
            return False
        with open(cache_path, "rb") as f:
            mtimes = pickle.load(f).get("file_mtimes", {})
        for fpath, mtime in mtimes.items():
            if not Path(fpath).exists() or Path(fpath).stat().st_mtime > mtime + 0.1:
                return False
        return True

    # ── Query ────────────────────────────────────────────────────

    def _find_relevant_files(self, question: str) -> set[str] | None:
        """Stage 1: find files whose title or nickname appears in the question.
        Returns None if no match (→ search all files)."""
        matches: set[str] = set()
        for fname, title in self._titles.items():
            if title and title in question:
                matches.add(fname)
        for fname, nicks in self._nicknames.items():
            for nick in nicks:
                if nick in question:
                    matches.add(fname)
                    break
        return matches if matches else None

    def ask(self, question: str, top_k: int = 14) -> str:
        """Ask a question about the loaded documents."""
        if not self._retriever or not self._chunks:
            return "No documents loaded. Call .load() or .load_cache() first."

        # Rewrite slang → formal for better retrieval
        search_query = self._rewrite_query(question)

        relevant = self._find_relevant_files(search_query) or self._find_relevant_files(question)
        if self._verbose:
            if relevant:
                print(f"[Title match] -> {relevant}")
            else:
                print(f"[Title match] no match, searching all files")

        results = self._retriever.search_diverse(
            search_query, self._sources, top_k=top_k, per_file=2,
            file_filter=relevant)
        context = "\n\n---\n\n".join(chunk for chunk, _ in results)

        if self._verbose:
            print(f"[DEBUG] Query: {question}")
            for i, (chunk, score) in enumerate(results):
                try:
                    idx = self._chunks.index(chunk)
                    src = self._sources[idx]
                except ValueError:
                    src = "?"
                print(f"[DEBUG]   [{i}] score={score:.4f}  file={src}  "
                      f"text={chunk[:80]}...")

        prompt = (
            "You are a helpful assistant. Answer the user's question based ONLY on "
            "the provided document excerpts. If the documents don't contain enough "
            "information, say so honestly.\n\n"
            f"### Documents\n{context}\n\n"
            f"### Question\n{question}\n\n"
            "### Answer"
        )

        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=1024,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.choices[0].message.content
            except APIConnectionError:
                if attempt < 2:
                    wait = (attempt + 1) * 2
                    if self._verbose:
                        print(f"[Retry] Connection error, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    raise


# ── Helpers ───────────────────────────────────────────────────────

def load_dotenv(path: str = ".env") -> dict[str, str]:
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
