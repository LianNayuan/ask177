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

    # CJK Unicode ranges (including extensions, punctuation excluded)
    _CJK_RE = re.compile(r"[一-鿿㐀-䶿豈-﫿]+")

    def _tokenize(self, text: str) -> list[str]:
        """Tokenize text: character bigrams for CJK, \\w+ for others."""
        tokens: list[str] = []
        pos = 0
        for m in self._CJK_RE.finditer(text):
            # Non-CJK part before this CJK span
            if m.start() > pos:
                tokens.extend(re.findall(r"\w+", text[pos:m.start()].lower()))
            # CJK span → character bigrams
            cjk = m.group()
            for i in range(len(cjk) - 1):
                tokens.append(cjk[i:i + 2])
            # Also add unigrams for single-char matches (names, etc.)
            tokens.extend(cjk)
            pos = m.end()
        # Remaining non-CJK tail
        if pos < len(text):
            tokens.extend(re.findall(r"\w+", text[pos:].lower()))
        return tokens

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
                       file_filter: set[str] | None = None,
                       dense_scores: list[float] | None = None,
                       dense_weight: float = 0.5) -> list[tuple[str, float]]:
        """Retrieve top chunks while ensuring each source file gets at least 1 slot.
        If file_filter is given, only consider chunks from those files.
        If dense_scores is provided, fuse with TF-IDF via weighted sum."""
        qv = self._query_vec(query)
        scored = [(c, self._dot(qv, dv), src)
                  for (c, dv), src in zip(zip(self.chunks, self._doc_vecs), sources)]

        # Hybrid fusion: min-max normalize both score sets, weighted sum
        if dense_scores is not None and len(dense_scores) == len(scored):
            max_tfidf = max(s for _, s, _ in scored) if scored else 1.0
            max_dense = max(dense_scores) if dense_scores else 1.0
            if max_tfidf == 0:
                max_tfidf = 1.0
            if max_dense == 0:
                max_dense = 1.0
            fused = []
            for i, (c, tfidf_s, src) in enumerate(scored):
                tfidf_norm = tfidf_s / max_tfidf
                dense_norm = dense_scores[i] / max_dense
                combined = (1 - dense_weight) * tfidf_norm + dense_weight * dense_norm
                fused.append((c, combined, src))
            scored = fused

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
            if file_counts[src] < per_file and (chunk, score) not in result:
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


# ── Dense Retriever ──────────────────────────────────────────────

class DenseRetriever:
    """Dense vector retriever using cosine similarity on pre-computed embeddings.
    Zero external dependencies — pure Python math."""

    def __init__(self, chunks: list[str], embeddings: list[list[float]]):
        self.chunks = chunks
        self._embeddings = [self._l2_normalize(e) for e in embeddings]
        self._dim = len(embeddings[0]) if embeddings else 0

    @staticmethod
    def _l2_normalize(vec: list[float]) -> list[float]:
        norm = math.sqrt(sum(x * x for x in vec))
        if norm == 0:
            return vec
        return [x / norm for x in vec]

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def score_all(self, query_embedding: list[float]) -> list[float]:
        """Return cosine similarity score for every chunk (for hybrid fusion)."""
        qv = self._l2_normalize(query_embedding)
        return [self._dot(qv, ev) for ev in self._embeddings]

    def search(self, query_embedding: list[float], top_k: int = 3
               ) -> list[tuple[str, float]]:
        """Top-k semantic search."""
        qv = self._l2_normalize(query_embedding)
        scored = [(c, self._dot(qv, ev)) for c, ev in zip(self.chunks, self._embeddings)]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def _state(self) -> dict:
        return {"embeddings": self._embeddings, "chunks": self.chunks}

    @classmethod
    def _from_state(cls, state: dict) -> "DenseRetriever":
        obj = cls.__new__(cls)
        obj._embeddings = state["embeddings"]
        obj.chunks = state["chunks"]
        obj._dim = len(obj._embeddings[0]) if obj._embeddings else 0
        return obj


# ── ChromaDB Retriever ────────────────────────────────────────────

class ChromaRetriever:
    """Dense retriever backed by ChromaDB for persistent vector storage."""

    def __init__(self, chunks: list[str] | None = None,
                 embeddings: list[list[float]] | None = None,
                 sources: list[str] | None = None,
                 persist_dir: str = "chroma_db",
                 collection_name: str = "weapons"):
        import chromadb
        self._persist_dir = persist_dir
        self._collection_name = collection_name
        self.chunks = chunks or []

        self._client = chromadb.PersistentClient(path=persist_dir)

        try:
            self._collection = self._client.get_collection(collection_name)
        except Exception:
            self._collection = self._client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            if chunks and embeddings:
                self._add_batch(chunks, embeddings, sources or [])

    def _add_batch(self, chunks: list[str], embeddings: list[list[float]],
                   sources: list[str], batch_size: int = 200):
        ids = [str(i) for i in range(len(chunks))]
        metas = [{"source": sources[i]} if i < len(sources) else {}
                 for i in range(len(chunks))]
        for i in range(0, len(chunks), batch_size):
            end = min(i + batch_size, len(chunks))
            self._collection.add(
                ids=ids[i:end],
                documents=chunks[i:end],
                embeddings=embeddings[i:end],
                metadatas=metas[i:end],
            )

    def score_all(self, query_embedding: list[float]) -> list[float]:
        """Return cosine similarity scores for all chunks (for hybrid fusion)."""
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=len(self.chunks),
            include=["distances"],
        )
        # cosine space: distance = 1 - similarity  →  similarity = 1 - distance
        distances = results["distances"][0]
        ids = results["ids"][0]
        scores = [0.0] * len(self.chunks)
        for doc_id, dist in zip(ids, distances):
            scores[int(doc_id)] = 1.0 - dist
        return scores

    def search(self, query_embedding: list[float], top_k: int = 3
               ) -> list[tuple[str, float]]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "distances"],
        )
        docs = results["documents"][0]
        distances = results["distances"][0]
        return [(doc, 1.0 - dist) for doc, dist in zip(docs, distances)]

    def _state(self) -> dict:
        return {
            "persist_dir": self._persist_dir,
            "collection_name": self._collection_name,
            "chunk_count": len(self.chunks),
        }

    @classmethod
    def _from_state(cls, state: dict, chunks: list[str] | None = None) -> "ChromaRetriever":
        obj = cls.__new__(cls)
        obj._persist_dir = state["persist_dir"]
        obj._collection_name = state["collection_name"]
        obj.chunks = chunks or []
        import chromadb
        obj._client = chromadb.PersistentClient(path=obj._persist_dir)
        obj._collection = obj._client.get_collection(obj._collection_name)
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
                 verbose: bool = False,
                 embedding_model: str | None = None,
                 embedding_api_key: str | None = None,
                 embedding_base_url: str | None = None,
                 dense_weight: float = 0.5,
                 chroma_db: str | None = None,
                 retrieval_mode: str = "hybrid"):
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

        # Dense retrieval
        self._embedding_model = embedding_model
        self._embedding_api_key = embedding_api_key or key
        self._embedding_base_url = embedding_base_url or self.DEEPSEEK_BASE
        self._dense_weight = dense_weight
        self._embeddings: list[list[float]] = []
        self._embedding_model_used: str = ""
        self._dense_retriever: DenseRetriever | None = None
        self._chroma_retriever: ChromaRetriever | None = None
        self._chroma_db: str | None = chroma_db
        if retrieval_mode not in ("tfidf", "dense", "hybrid"):
            raise ValueError(f"retrieval_mode must be 'tfidf', 'dense', or 'hybrid', got: {retrieval_mode!r}")
        self._retrieval_mode: str = retrieval_mode
        self._embedding_client: OpenAI | None = None
        self._local_embedding_model: object | None = None  # lazily loaded SentenceTransformer
        self._last_query_info: dict = {}  # metadata about the most recent query

    # ── Logging ──────────────────────────────────────────────────

    def _log(self, msg: str):
        if self._verbose:
            ts = time.strftime("%H:%M:%S")
            try:
                print(f"[{ts}] {msg}")
            except UnicodeEncodeError:
                print(f"[{ts}] {msg.encode('ascii', errors='replace').decode('ascii')}")

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
            text = self._sanitize(fp.read_text(encoding="utf-8", errors="replace"))
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
        self._merge_table_nicknames()

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
        if self._glossary:
            self._log(f"[Glossary] Loaded {len(self._glossary)} mappings")

    def _rewrite_query(self, question: str) -> str:
        """Replace slang terms in the question with formal equivalents."""
        result = question
        for slang, formal in self._glossary.items():
            if slang in result:
                result = result.replace(slang, formal)
                self._log(f"[Rewrite] '{slang}' → '{formal}'")
        if result != question:
            self._log(f"[Rewrite] 最终: {question!r} → {result!r}")
        return result

    # ── Dense embeddings ─────────────────────────────────────────

    def _get_embedding_client(self) -> OpenAI:
        if self._embedding_client is None:
            self._embedding_client = OpenAI(
                api_key=self._embedding_api_key,
                base_url=self._embedding_base_url)
        return self._embedding_client

    def _load_local_embedding_model(self) -> object | None:
        """Lazy-load the local SentenceTransformer model. Returns None on failure."""
        if self._local_embedding_model is not None:
            return self._local_embedding_model
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            self._log("[Embed] sentence-transformers not installed, cannot use local model")
            return None
        try:
            self._log(f"[Embed] Loading local model: {self._embedding_model}")
            self._local_embedding_model = SentenceTransformer(self._embedding_model)
            return self._local_embedding_model
        except Exception as e:
            self._log(f"[Embed] Failed to load local model: {e}")
            return None

    def _is_deepseek_model(self) -> bool:
        return bool(self._embedding_model and self._embedding_model.startswith("deepseek-"))

    def _embed_query(self, query: str) -> list[float] | None:
        """Get dense embedding for a single query. Returns None on failure."""
        if not self._embedding_model:
            return None

        if self._is_deepseek_model():
            return self._embed_query_api(query)
        else:
            return self._embed_query_local(query)

    def _embed_query_api(self, query: str) -> list[float] | None:
        try:
            client = self._get_embedding_client()
            resp = client.embeddings.create(
                model=self._embedding_model,
                input=query,
            )
            return resp.data[0].embedding
        except Exception as e:
            self._log(f"[Embed] API query embedding failed: {e}")
            return None

    def _embed_query_local(self, query: str) -> list[float] | None:
        model = self._load_local_embedding_model()
        if model is None:
            return None
        try:
            emb = model.encode([query], normalize_embeddings=True, show_progress_bar=False)
            return emb[0].tolist() if hasattr(emb, "tolist") else list(emb[0])
        except Exception as e:
            self._log(f"[Embed] Local query embedding failed: {e}")
            return None

    def _embed_chunks(self, chunks: list[str],
                      batch_size: int = 100) -> list[list[float]]:
        """Embed all chunks via API (build-time only, not used at query time)."""
        client = self._get_embedding_client()
        all_embeddings: list[list[float]] = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            resp = client.embeddings.create(
                model=self._embedding_model,
                input=batch,
            )
            all_embeddings.extend([d.embedding for d in resp.data])
            self._log(f"[Embed] {min(i + batch_size, len(chunks))}/{len(chunks)} chunks embedded")
        return all_embeddings

    @staticmethod
    def _sanitize(text: str) -> str:
        """Remove lone surrogates that break JSON encoding."""
        return ''.join(c for c in text if not ('\ud800' <= c <= '\udfff'))

    @staticmethod
    def _extract_title(text: str) -> str | None:
        """Extract the first # heading, trimmed to the part before ' - '."""
        m = re.search(r"^#\s+(.+)", text, re.MULTILINE)
        if not m:
            return None
        title = m.group(1).strip()
        if " - " in title:
            title = re.sub(r"\s*[-—–]\s*(完整)?武器详情.*$", "", title)
        return title

    @staticmethod
    def _extract_nicknames(text: str) -> list[str]:
        """Extract nicknames from the markdown.

        For Chinese files: parses '俗称：红牙刷、蓝牙刷'.
        For English files: parses '## Names in Other Languages' for
        Chinese/Japanese names as cross-language lookup keys.
        """
        # Try Chinese nickname field first
        m = re.search(r"俗称[：:]\s*(.+)", text)
        if m:
            raw = m.group(1).strip()
            if raw not in ("—", "—", "-", "无", ""):
                parts = re.split(r"[、，,，]", raw)
                return [p.strip() for p in parts if p.strip()]

        # For English files: extract Chinese/Japanese names from
        # "## Names in Other Languages" section
        names_section = re.search(
            r'## Names in Other Languages\s*\n+(.*?)(?=\n## |\Z)',
            text, re.DOTALL
        )
        if not names_section:
            return []

        nicks: list[str] = []
        for m in re.finditer(
            r'-\s*\*\*(Chinese|Japanese|Korean)\b[^:]*\*\*\s*:\s*(.+?)(?:\n|$)',
            names_section.group(1), re.IGNORECASE
        ):
            raw_value = m.group(2).strip()
            parts = raw_value.split()
            name_parts: list[str] = []
            for p in parts:
                if re.match(r'^[a-zA-Z]', p):
                    break
                if p.startswith('('):
                    break
                name_parts.append(p)
            name = ' '.join(name_parts).strip()
            if name and len(name) >= 2:
                nicks.append(name)

        return nicks

    @staticmethod
    def _parse_nickname_table(text: str) -> dict[str, list[str]]:
        """Parse a markdown table (官中译名 | 俗称 | ...) → {official: [nicknames]}."""
        result: dict[str, list[str]] = {}
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith("|--") or line.startswith("|---"):
                continue
            parts = [p.strip() for p in line.split("|")]
            # Table row: | col1 | col2 | col3 |
            if len(parts) < 4:
                continue
            official = parts[1]
            nickname = parts[2]
            if not official or not nickname:
                continue
            if official in ("官中译名", "官中译名"):
                continue
            nicks = [n.strip() for n in re.split(r"\s*/\s*", nickname) if n.strip()]
            if official not in result:
                result[official] = []
            for n in nicks:
                if n not in result[official]:
                    result[official].append(n)
        return result

    def _merge_table_nicknames(self) -> int:
        """If 武器俗称及来源.md was loaded, parse its table and merge nicknames.
        Returns number of newly added nicknames."""
        # Find the table file among loaded files

        table_fp = None
        for fp_str in self._file_mtimes:
            if Path(fp_str).name == "武器俗称及来源.md":
                table_fp = Path(fp_str)
                break
        if not table_fp or not table_fp.is_file():
            self._log("[Nickname table] 武器俗称及来源.md not found in loaded files, skip")
            return 0

        table_data = self._parse_nickname_table(
            self._sanitize(table_fp.read_text(encoding="utf-8", errors="replace")))
        if not table_data:
            self._log("[Nickname table] Table parsed but no data extracted")
            return 0

        title_to_fname = {title: fname for fname, title in self._titles.items()}
        merged = 0
        skipped = 0
        for official, nicks in table_data.items():
            fname = title_to_fname.get(official)
            if not fname:
                self._log(f"[Nickname table] SKIP '{official}': no matching file title in {sorted(self._titles.values())[:5]}...")
                skipped += 1
                continue
            existing = set(self._nicknames.get(fname, []))
            new = [n for n in nicks if n not in existing]
            if new:
                self._nicknames[fname] = list(existing | set(nicks))
                self._log(f"[Nickname table] '{fname}' +{new} (was {list(existing)})")
                merged += len(new)
        if self._verbose:
            self._log(f"[Nickname table] Merged {merged} new nicknames,"
                      f" {skipped} table rows unmatched, {len(table_data)} total rows")
        else:
            print(f"[Nickname table] Merged {merged} new nicknames"
                  f" ({skipped} rows unmatched, {len(table_data)} total)")
        return merged

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

    # ── Incremental update ───────────────────────────────────────

    def incremental_update(self, *paths: str, chunk_size: int = 500,
                           overlap: int = 100) -> bool:
        """Update only changed/new/deleted files, then rebuild retriever.
        Returns True if anything was updated, False if nothing changed."""
        # 1. Discover current files on disk
        current: dict[str, Path] = {}
        for path in paths:
            p = Path(path)
            if p.is_dir():
                for fp in sorted(p.rglob("*.md")):
                    current[str(fp)] = fp
            elif p.suffix == ".md":
                current[str(p)] = p

        old_paths = set(self._file_mtimes.keys())
        new_paths = set(current.keys())

        added = new_paths - old_paths
        deleted = old_paths - new_paths
        modified = {fp for fp in (new_paths & old_paths)
                    if current[fp].stat().st_mtime > self._file_mtimes[fp] + 0.1}

        if not (added or deleted or modified):
            return self._merge_table_nicknames() > 0  # True only if table added nicknames

        # 2. Remove chunks from deleted and modified files
        removed_names = {Path(fp).name for fp in (deleted | modified)}
        if removed_names:
            keep = [(c, s) for c, s in zip(self._chunks, self._sources)
                    if s not in removed_names]
            self._chunks = [c for c, _ in keep]
            self._sources = [s for _, s in keep]
            self._file_names = [n for n in self._file_names if n not in removed_names]
            for fp in deleted | modified:
                self._file_mtimes.pop(fp, None)
                name = Path(fp).name
                self._titles.pop(name, None)
                self._nicknames.pop(name, None)

        # 3. Add/re-add chunks from added and modified files
        for fp_str in sorted(added | modified):
            fp = current[fp_str]
            self._file_mtimes[fp_str] = fp.stat().st_mtime
            text = self._sanitize(fp.read_text(encoding="utf-8", errors="replace"))
            if fp.name not in self._file_names:
                self._file_names.append(fp.name)
            title = self._extract_title(text) or fp.stem
            self._titles[fp.name] = title
            nicks = self._extract_nicknames(text)
            if nicks:
                self._nicknames[fp.name] = nicks
            else:
                self._nicknames.pop(fp.name, None)
            new_chunks = self._chunk(text, chunk_size, overlap)
            prefixed = [f"[{title}]\n{c}" for c in new_chunks]
            self._chunks.extend(prefixed)
            self._sources.extend([fp.name] * len(prefixed))

        # 4. Rebuild TF-IDF from all chunks
        self._retriever = TfidfRetriever(self._chunks)
        self._merge_table_nicknames()

        self._log(f"[Incremental] +{len(added)} ~{len(modified)} -{len(deleted)} files, "
                  f"{len(self._chunks)} chunks total")
        return True

    # ── Cache ────────────────────────────────────────────────────

    def save_cache(self, path: str = "index.pkl", db=None):
        """Save chunks and retriever state.

        If db is provided, structured data (chunks, sources, file metadata,
        titles, nicknames) goes to SQLite; pickle only holds retriever state
        and dense config.  Without db, everything goes into pickle (legacy)."""
        if not self._retriever:
            raise RuntimeError("No index built. Call .load() first.")

        if db:
            meta = {}
            if self._embedding_model_used:
                meta["embedding_model_used"] = self._embedding_model_used
            if self._chroma_db:
                meta["chroma_db"] = self._chroma_db
            db.save_knowledge(
                self._file_names, self._file_mtimes, self._titles,
                self._chunks, self._sources, self._nicknames,
                meta=meta if meta else None)
            data = {
                "retriever": self._retriever._state(),
                "embeddings": self._embeddings,
            }
        else:
            data = {
                "chunks": self._chunks,
                "sources": self._sources,
                "file_names": self._file_names,
                "file_mtimes": self._file_mtimes,
                "titles": self._titles,
                "nicknames": self._nicknames,
                "glossary": self._glossary,
                "retriever": self._retriever._state(),
                "embeddings": self._embeddings,
                "embedding_model_used": self._embedding_model_used,
                "chroma_db": self._chroma_db,
            }

        with open(path, "wb") as f:
            pickle.dump(data, f)

        if self._chroma_db:
            emb_info = f", chromadb://{self._chroma_db}"
        elif self._embeddings:
            emb_info = f", {len(self._embeddings)} dense vectors"
        else:
            emb_info = ", NO dense embeddings"
        db_info = " + SQLite" if db else ""
        self._log(f"[Cache] Saved {len(self._chunks)} chunks{emb_info} → {path}{db_info}")

    def load_cache(self, path: str = "index.pkl", force: bool = False,
                   db=None) -> bool:
        """Load cached index. If db is provided, structured metadata comes from
        SQLite and only the retriever + dense config are read from pickle.
        Without db, everything is read from pickle (legacy / deployment mode)."""
        pkl_path = Path(path)
        have_pkl = pkl_path.exists()
        have_db = db and db.has_knowledge()

        if not have_pkl and not have_db:
            self._log(f"[Cache] No cache found (no {path}, no knowledge in DB)")
            return False

        # ── Structured data: prefer SQLite, fall back to pickle ─────
        if have_db:
            k = db.load_knowledge()
            self._chunks = k["chunks"]
            self._sources = k["sources"]
            self._file_names = k["file_names"]
            self._file_mtimes = k["file_mtimes"]
            self._titles = k["titles"]
            self._nicknames = k["nicknames"]
            self._embedding_model_used = k.get("meta", {}).get("embedding_model_used", "")
            self._chroma_db = k.get("meta", {}).get("chroma_db")
            self._glossary = db.load_glossary()
        elif have_pkl:
            with open(path, "rb") as f:
                data = pickle.load(f)
            self._chunks = data["chunks"]
            self._sources = data["sources"]
            self._file_names = data["file_names"]
            self._file_mtimes = data["file_mtimes"]
            self._titles = data.get("titles", {})
            self._nicknames = data.get("nicknames", {})
            self._glossary = data.get("glossary", {})
        else:
            return False

        # ── Freshness check ─────────────────────────────────────────
        if not force:
            stale = []
            for fpath, mtime in self._file_mtimes.items():
                if not Path(fpath).exists() or Path(fpath).stat().st_mtime > mtime + 0.1:
                    stale.append(fpath)
            if stale:
                self._log(f"[Cache] Stale — {len(stale)} file(s) changed: {stale[:3]}...")
                return False

        # ── Retriever always from pickle ───────────────────────────
        if have_pkl:
            if have_db:
                with open(path, "rb") as f:
                    data = pickle.load(f)
                # Migration: if DB meta doesn't have chroma config yet,
                # fall back to legacy pickle values
                if not self._chroma_db:
                    self._chroma_db = data.get("chroma_db")
                if not self._embedding_model_used:
                    self._embedding_model_used = data.get("embedding_model_used", "")
            self._retriever = TfidfRetriever._from_state(data["retriever"])
            self._embeddings = data.get("embeddings", [])
            if not have_db:
                self._embedding_model_used = data.get("embedding_model_used", "")
                self._chroma_db = data.get("chroma_db")
        else:
            self._retriever = TfidfRetriever(self._chunks)

        # ── ChromaDB / Dense ────────────────────────────────────────
        if self._chroma_db:
            try:
                self._chroma_retriever = ChromaRetriever._from_state(
                    {"persist_dir": self._chroma_db,
                     "collection_name": "weapons",
                     "chunk_count": len(self._chunks)},
                    chunks=self._chunks)
                if not self._embedding_model:
                    self._embedding_model = self._embedding_model_used
                self._log(f"[Cache] Loaded {len(self._chunks)} chunks,"
                          f" ChromaDB ({self._chroma_db})"
                          f" ({self._embedding_model_used})"
                          f"{' + SQLite' if have_db else ' from ' + path}")
            except Exception as e:
                self._log(f"[Cache] ChromaDB connect failed: {e},"
                          f" falling back to in-memory")
                self._chroma_db = None
        elif self._embeddings:
            self._dense_retriever = DenseRetriever(self._chunks, self._embeddings)
            if not self._embedding_model:
                self._embedding_model = self._embedding_model_used

        if not self._chroma_db:
            if self._embeddings:
                self._log(f"[Cache] Loaded {len(self._chunks)} chunks,"
                          f" {len(self._embeddings)} dense vectors"
                          f" ({self._embedding_model_used})"
                          f"{' + SQLite' if have_db else ' from ' + path}")
            else:
                self._log(f"[Cache] Loaded {len(self._chunks)} chunks,"
                          f" NO dense embeddings"
                          f"{' + SQLite' if have_db else ' from ' + path}")

        # Eagerly preload local embedding model (skip when TF-IDF only)
        if self._retrieval_mode != "tfidf" \
                and (self._chroma_retriever or self._dense_retriever) \
                and self._embedding_model \
                and not self._is_deepseek_model():
            self._log(f"Preloading embedding model: {self._embedding_model}")
            self._load_local_embedding_model()

        return True

    def is_fresh(self, cache_path: str = "index.pkl", db=None) -> bool:
        """Check if cache is up-to-date without fully loading."""
        if db and db.has_knowledge():
            k = db.load_knowledge()
            for fpath, mtime in k.get("file_mtimes", {}).items():
                if not Path(fpath).exists() or Path(fpath).stat().st_mtime > mtime + 0.1:
                    return False
            return Path(cache_path).exists()  # retriever pickle must also exist
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
        q_lower = question.lower()
        for fname, title in self._titles.items():
            if title and title.lower() in q_lower:
                self._log(f"[Title match] title '{title}' in question → {fname}")
                matches.add(fname)
        for fname, nicks in self._nicknames.items():
            for nick in nicks:
                if nick.lower() in q_lower:
                    self._log(f"[Title match] nickname '{nick}' in question → {fname}")
                    matches.add(fname)
                    break
        return matches if matches else None

    def _rewrite_with_context(self, question: str,
                               history: list[dict[str, str]] | None) -> str:
        """Use the LLM to rewrite the question into an optimized search query,
        resolving pronouns, combining clarifications, and expanding abbreviations
        based on conversation context.

        Returns the rewritten query, or the original question on failure."""
        if not history:
            return question

        recent = history[-10:] if len(history) > 10 else history
        history_text = "\n".join(
            f"{'User' if h['role'] == 'user' else 'Assistant'}: {h['content'][:200]}"
            for h in recent)

        prompt = (
            "Given this conversation, rewrite the user's LATEST message into a "
            "self-contained search query for a weapon knowledge base.\n\n"
            "Rules:\n"
            "- If the user is clarifying or correcting (e.g. '是4k', '第一个'), "
            "combine it with the previous question's topic. Example: '是4k' + "
            "previous '开开的大招是什么' → '4K 特殊武器' or '公升4K 大招'.\n"
            "- Resolve pronouns (它的/这个/那个) to the actual weapon name.\n"
            "- Keep the user's original intent even if you add weapon names.\n"
            "- Output ONLY the rewritten query, nothing else.\n\n"
            f"Conversation:\n{history_text}\n\n"
            f"Latest user message: \"{question}\"\n"
            "Search query:"
        )

        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                max_tokens=80,
                messages=[{"role": "user", "content": prompt}],
            )
            result = resp.choices[0].message.content.strip()
            # Strip any quotes the model might add
            result = result.strip('"\'')
            if result and result != question:
                self._log(f"[Rewrite] context-aware: {question!r} → {result!r}")
                return result
            return question
        except Exception as e:
            self._log(f"[Rewrite] context rewrite failed ({e}), using original")
            return question

    def ask(self, question: str, top_k: int = 14, debug_chunks: bool = False,
            history: list[dict[str, str]] | None = None) -> str:
        """Ask a question about the loaded documents.
        history: prior turns in this conversation (role/content dicts)."""
        if not self._retriever or not self._chunks:
            self._last_query_info = {
                "question": question, "search_query": question,
                "rewrite": "", "hit_files": "", "mode": "error",
                "dense_source": None, "error": "No documents loaded",
            }
            return "No documents loaded. Call .load() or .load_cache() first."

        # ── Stage 1: Query rewriting ──────────────────────────────
        # Let the LLM resolve pronouns, combine clarifications, and expand
        # abbreviations using conversation context.
        search_query = self._rewrite_with_context(question, history)
        # Then apply glossary rules (lightweight, no API call)
        search_query = self._rewrite_query(search_query)

        # ── Stage 2: File matching & retrieval ────────────────────
        relevant = self._find_relevant_files(search_query) or self._find_relevant_files(question)

        # Always include the nickname reference file
        for ref in self._file_names:
            if ref == "武器俗称及来源.md" and (relevant is None or ref not in relevant):
                if relevant is None:
                    relevant = set()
                relevant.add(ref)
                self._log(f"[Title match] also including reference file: {ref}")

        if relevant:
            self._log(f"[Title match] -> {relevant}")
        else:
            self._log(f"[Title match] no match, searching all files")

        # Get dense embedding scores based on retrieval mode
        dense_scores = None
        dense_source: str | None = None
        dense_weight = self._dense_weight

        if self._retrieval_mode == "tfidf":
            self._log("[Retrieval] mode=tfidf, using TF-IDF only")
        elif self._retrieval_mode in ("dense", "hybrid"):
            if self._chroma_retriever and self._embedding_model:
                query_emb = self._embed_query(search_query)
                if query_emb is not None:
                    dense_scores = self._chroma_retriever.score_all(query_emb)
                    dense_source = "chroma"
            elif self._dense_retriever and self._embedding_model:
                query_emb = self._embed_query(search_query)
                if query_emb is not None:
                    dense_scores = self._dense_retriever.score_all(query_emb)
                    dense_source = "memory"

            if self._retrieval_mode == "dense":
                if dense_scores is None:
                    self._last_query_info = {
                        "question": question_sanitized, "search_query": search_query,
                        "rewrite": "", "hit_files": "", "mode": "Dense only (error)",
                        "dense_source": None, "error": "No dense embeddings available",
                    }
                    return ("Error: --mode dense requires dense embeddings, "
                            "but none are available. Run: python build_embeddings.py --chroma")
                dense_weight = 1.0
                self._log("[Retrieval] mode=dense, using dense vectors only")
            else:
                self._log("[Retrieval] mode=hybrid")

        if dense_scores is not None:
            tag = f"[Dense/{dense_source}]"
            self._log(f"{tag} using dense scores, model={self._embedding_model}, weight={dense_weight}")
            dense_top = sorted(enumerate(dense_scores), key=lambda x: x[1], reverse=True)
            self._log(f"{tag} top-5 hits:")
            for rank, (idx, score) in enumerate(dense_top[:5]):
                src = self._sources[idx] if idx < len(self._sources) else "?"
                snippet = self._chunks[idx][:80].replace("\n", " ") if idx < len(self._chunks) else "?"
                self._log(f"{tag}   [{rank}] score={score:.4f}  file={src}")
                self._log(f"{tag}        text: {snippet}...")
        else:
            if self._retrieval_mode == "tfidf":
                self._log("[Dense] TF-IDF only mode (dense disabled by --mode)")
            elif self._chroma_retriever is None and self._dense_retriever is None:
                self._log("[Dense] no retriever loaded, using TF-IDF only")
            elif not self._embedding_model:
                self._log("[Dense] no embedding model configured, using TF-IDF only")
            else:
                self._log("[Dense] query embedding failed, falling back to TF-IDF only")

        results = self._retriever.search_diverse(
            search_query, self._sources, top_k=top_k, per_file=2,
            file_filter=relevant,
            dense_scores=dense_scores,
            dense_weight=dense_weight)

        # ── Stage 3: Build context ────────────────────────────────
        context = "\n\n---\n\n".join(chunk for chunk, _ in results)

        if debug_chunks:
            print(f"[DEBUG] Query: {question}")
            for i, (chunk, score) in enumerate(results):
                try:
                    idx = self._chunks.index(chunk)
                    src = self._sources[idx]
                except ValueError:
                    src = "?"
                print(f"[DEBUG]   [{i}] score={score:.4f}  file={src}  "
                      f"text={chunk[:80]}...")

        context = self._sanitize(context)
        question_sanitized = self._sanitize(question)

        # ── Stage 4: Build messages & call LLM ────────────────────
        system_prompt = (
            "You are a helpful assistant. Answer the user's question based ONLY on "
            "the provided document excerpts.\n"
            "If the question is vague or the documents lack key details (e.g. the "
            "user asks 'tell me about X' but doesn't specify which aspect), ask a "
            "brief clarifying question to narrow down what they want.\n"
            "If the documents genuinely don't contain the answer even after "
            "clarification, say so honestly."
        )
        user_content = (f"### Documents\n{context}\n\n"
                        f"### Question\n{question_sanitized}")

        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        if history:
            for h in history:
                messages.append({"role": h["role"],
                                 "content": self._sanitize(h["content"])})
        messages.append({"role": "user", "content": user_content})

        # Collect query metadata for logging
        result_files = {self._sources[self._chunks.index(c)]
                        for c, _ in results if c in self._chunks}
        if self._retrieval_mode == "tfidf":
            ret_mode = "TF-IDF only"
        elif self._retrieval_mode == "dense":
            ret_mode = f"Dense only ({dense_source or 'unknown'})"
        elif self._chroma_retriever:
            ret_mode = "TF-IDF + Dense/chroma"
        elif self._dense_retriever:
            ret_mode = "TF-IDF + Dense/memory"
        else:
            ret_mode = "TF-IDF only"
        self._last_query_info = {
            "question": question_sanitized,
            "search_query": search_query,
            "rewrite": question_sanitized if question_sanitized != search_query else "",
            "hit_files": ", ".join(sorted(result_files)),
            "mode": ret_mode,
            "dense_source": dense_source,
            "error": "",
        }

        for attempt in range(3):
            try:
                resp = self._client.chat.completions.create(
                    model=self._model,
                    max_tokens=1024,
                    messages=messages,
                )
                answer = resp.choices[0].message.content
                self._last_query_info["answer"] = answer
                return answer
            except APIConnectionError as e:
                if attempt < 2:
                    wait = (attempt + 1) * 2
                    self._log(f"[Retry] Connection error, waiting {wait}s...")
                    time.sleep(wait)
                else:
                    self._last_query_info["error"] = f"API connection failed after 3 retries: {e}"
                    raise
            except Exception as e:
                self._last_query_info["error"] = f"LLM call failed: {e}"
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
