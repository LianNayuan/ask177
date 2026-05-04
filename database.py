"""SQLite database for query logs, glossary, and knowledge metadata."""

import sqlite3
import time
from pathlib import Path

DB_PATH = "data.db"


class Database:
    def __init__(self, path: str = DB_PATH):
        self._path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_tables()

    def _init_tables(self):
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS query_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                question    TEXT NOT NULL,
                answer      TEXT NOT NULL,
                mode        TEXT,
                hit_files   TEXT,
                rewrite     TEXT,
                latency_ms  INTEGER,
                session_id  INTEGER DEFAULT 0,
                error       TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS glossary (
                slang       TEXT PRIMARY KEY,
                formal      TEXT NOT NULL,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS feedback (
                query_id    INTEGER PRIMARY KEY,
                rating      INTEGER NOT NULL CHECK(rating BETWEEN 1 AND 5),
                comment     TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS knowledge_files (
                filename    TEXT PRIMARY KEY,
                filepath    TEXT NOT NULL,
                mtime       REAL NOT NULL,
                title       TEXT
            );

            CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                chunk_text  TEXT NOT NULL,
                source      TEXT NOT NULL,
                chunk_order INTEGER NOT NULL
            );

            CREATE TABLE IF NOT EXISTS knowledge_nicknames (
                filename    TEXT NOT NULL,
                nickname    TEXT NOT NULL,
                PRIMARY KEY (filename, nickname)
            );

            CREATE TABLE IF NOT EXISTS knowledge_meta (
                key         TEXT PRIMARY KEY,
                value       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT,
                created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                role            TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content         TEXT NOT NULL,
                hit_files       TEXT,
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        # Migrations: add columns that may not exist in older databases
        for col, col_def in [("session_id", "INTEGER DEFAULT 0"),
                             ("error", "TEXT")]:
            try:
                self._conn.execute(f"ALTER TABLE query_logs ADD COLUMN {col} {col_def}")
            except sqlite3.OperationalError:
                pass  # column already exists
        self._conn.commit()

    # ── Query logs ──────────────────────────────────────────────────

    def log_query(self, question: str, answer: str, mode: str = "",
                  hit_files: str = "", rewrite: str = "", latency_ms: int = 0,
                  session_id: int = 0, error: str = "") -> int:
        cur = self._conn.execute(
            "INSERT INTO query_logs (question, answer, mode, hit_files, rewrite,"
            " latency_ms, session_id, error)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (question, answer, mode, hit_files, rewrite, latency_ms,
             session_id, error))
        self._conn.commit()
        return cur.lastrowid

    def recent_queries(self, limit: int = 20) -> list[sqlite3.Row]:
        return self._conn.execute(
            "SELECT id, question, mode, latency_ms, created_at"
            " FROM query_logs ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()

    def get_query(self, query_id: int) -> sqlite3.Row | None:
        return self._conn.execute(
            "SELECT * FROM query_logs WHERE id = ?", (query_id,)).fetchone()

    # ── Glossary ────────────────────────────────────────────────────

    def load_glossary(self) -> dict[str, str]:
        rows = self._conn.execute("SELECT slang, formal FROM glossary").fetchall()
        return {r["slang"]: r["formal"] for r in rows}

    def add_glossary(self, slang: str, formal: str):
        self._conn.execute(
            "INSERT OR REPLACE INTO glossary (slang, formal) VALUES (?, ?)",
            (slang, formal))
        self._conn.commit()

    def del_glossary(self, slang: str) -> bool:
        cur = self._conn.execute("DELETE FROM glossary WHERE slang = ?", (slang,))
        self._conn.commit()
        return cur.rowcount > 0

    def import_glossary_from_file(self, path: str) -> int:
        """Import slang|formal pairs from a file. Returns count added."""
        p = Path(path)
        if not p.is_file():
            return 0
        count = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                slang, formal = line.split("|", 1)
                self.add_glossary(slang.strip(), formal.strip())
                count += 1
        return count

    # ── Feedback ────────────────────────────────────────────────────

    def add_feedback(self, query_id: int, rating: int, comment: str = ""):
        self._conn.execute(
            "INSERT OR REPLACE INTO feedback (query_id, rating, comment)"
            " VALUES (?, ?, ?)", (query_id, rating, comment))
        self._conn.commit()

    # ── Stats ───────────────────────────────────────────────────────

    def stats(self) -> dict:
        total = self._conn.execute(
            "SELECT COUNT(*) as n FROM query_logs").fetchone()["n"]
        avg_latency = self._conn.execute(
            "SELECT AVG(latency_ms) as n FROM query_logs").fetchone()["n"]
        errors = self._conn.execute(
            "SELECT COUNT(*) as n FROM query_logs WHERE error IS NOT NULL AND error != ''"
        ).fetchone()["n"]
        sessions = self._conn.execute(
            "SELECT COUNT(DISTINCT session_id) as n FROM query_logs WHERE session_id > 0"
        ).fetchone()["n"]
        modes = {}
        for r in self._conn.execute(
                "SELECT mode, COUNT(*) as n FROM query_logs GROUP BY mode"):
            modes[r["mode"]] = r["n"]
        glossary_count = self._conn.execute(
            "SELECT COUNT(*) as n FROM glossary").fetchone()["n"]
        return {
            "total_queries": total,
            "avg_latency_ms": round(avg_latency or 0),
            "error_count": errors,
            "session_count": sessions,
            "retrieval_modes": modes,
            "glossary_entries": glossary_count,
        }

    # ── Knowledge metadata ──────────────────────────────────────────

    def save_knowledge(self, file_names: list[str],
                       file_mtimes: dict[str, float],
                       titles: dict[str, str],
                       chunks: list[str],
                       sources: list[str],
                       nicknames: dict[str, list[str]],
                       meta: dict[str, str] | None = None):
        """Replace all knowledge metadata in a single transaction."""
        with self._conn:
            self._conn.execute("DELETE FROM knowledge_chunks")
            self._conn.execute("DELETE FROM knowledge_nicknames")
            self._conn.execute("DELETE FROM knowledge_files")
            if meta is not None:
                for k, v in meta.items():
                    self._conn.execute(
                        "INSERT OR REPLACE INTO knowledge_meta (key, value)"
                        " VALUES (?, ?)", (k, v))

            for fname in file_names:
                # Find the full filepath from file_mtimes
                fp = ""
                for k in file_mtimes:
                    if Path(k).name == fname:
                        fp = k
                        break
                mtime = file_mtimes.get(fp, 0.0)
                title = titles.get(fname, "")
                self._conn.execute(
                    "INSERT INTO knowledge_files (filename, filepath, mtime, title)"
                    " VALUES (?, ?, ?, ?)",
                    (fname, fp, mtime, title))

            for i, (chunk, src) in enumerate(zip(chunks, sources)):
                self._conn.execute(
                    "INSERT INTO knowledge_chunks (chunk_text, source, chunk_order)"
                    " VALUES (?, ?, ?)",
                    (chunk, src, i))

            for fname, nicks in nicknames.items():
                for nick in nicks:
                    self._conn.execute(
                        "INSERT OR IGNORE INTO knowledge_nicknames (filename, nickname)"
                        " VALUES (?, ?)",
                        (fname, nick))

    def load_knowledge(self) -> dict:
        """Load knowledge metadata. Returns dict with keys:
        file_names, file_mtimes, titles, chunks, sources, nicknames.
        Returns empty dict if no knowledge data exists."""
        files = self._conn.execute(
            "SELECT filename, filepath, mtime, title FROM knowledge_files"
            " ORDER BY filename").fetchall()
        if not files:
            return {}

        file_names = [r["filename"] for r in files]
        file_mtimes = {r["filepath"]: r["mtime"] for r in files if r["filepath"]}
        titles = {r["filename"]: r["title"] for r in files if r["title"]}

        chunk_rows = self._conn.execute(
            "SELECT chunk_text, source FROM knowledge_chunks"
            " ORDER BY chunk_order").fetchall()
        chunks = [r["chunk_text"] for r in chunk_rows]
        sources = [r["source"] for r in chunk_rows]

        nick_rows = self._conn.execute(
            "SELECT filename, nickname FROM knowledge_nicknames").fetchall()
        nicknames: dict[str, list[str]] = {}
        for r in nick_rows:
            nicknames.setdefault(r["filename"], []).append(r["nickname"])

        meta_rows = self._conn.execute(
            "SELECT key, value FROM knowledge_meta").fetchall()
        meta = {r["key"]: r["value"] for r in meta_rows}

        return {
            "file_names": file_names,
            "file_mtimes": file_mtimes,
            "titles": titles,
            "chunks": chunks,
            "sources": sources,
            "nicknames": nicknames,
            "meta": meta,
        }

    def get_meta(self, key: str, default: str = "") -> str:
        row = self._conn.execute(
            "SELECT value FROM knowledge_meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def clear_knowledge(self):
        """Remove all knowledge metadata (for rebuild)."""
        self._conn.executescript("""
            DELETE FROM knowledge_chunks;
            DELETE FROM knowledge_nicknames;
            DELETE FROM knowledge_files;
            DELETE FROM knowledge_meta;
        """)
        self._conn.commit()

    def has_knowledge(self) -> bool:
        """Check if knowledge data exists in DB."""
        row = self._conn.execute(
            "SELECT COUNT(*) as n FROM knowledge_files").fetchone()
        return row["n"] > 0

    # ── Conversations ──────────────────────────────────────────────

    def create_conversation(self, title: str = "") -> int:
        """Create a new conversation. Returns its id."""
        cur = self._conn.execute(
            "INSERT INTO conversations (title) VALUES (?)", (title,))
        self._conn.commit()
        return cur.lastrowid

    def add_message(self, conversation_id: int, role: str, content: str,
                    hit_files: str = "") -> int:
        """Add a message to a conversation. Updates conversations.updated_at."""
        cur = self._conn.execute(
            "INSERT INTO messages (conversation_id, role, content, hit_files)"
            " VALUES (?, ?, ?, ?)",
            (conversation_id, role, content, hit_files))
        self._conn.execute(
            "UPDATE conversations SET updated_at = CURRENT_TIMESTAMP"
            " WHERE id = ?", (conversation_id,))
        # Auto-set title from first user message
        self._conn.execute(
            "UPDATE conversations SET title = ?"
            " WHERE id = ? AND title = ''",
            (content[:40], conversation_id))
        self._conn.commit()
        return cur.lastrowid

    def get_history(self, conversation_id: int, limit: int = 10
                    ) -> list[sqlite3.Row]:
        """Get recent messages from a conversation (newest last)."""
        return self._conn.execute(
            "SELECT role, content FROM messages"
            " WHERE conversation_id = ?"
            " ORDER BY id DESC LIMIT ?",
            (conversation_id, limit)).fetchall()[::-1]

    def recent_conversations(self, limit: int = 10) -> list[sqlite3.Row]:
        """List recent conversations."""
        return self._conn.execute(
            "SELECT c.id, c.title, c.created_at, c.updated_at,"
            " (SELECT COUNT(*) FROM messages WHERE conversation_id = c.id) as msg_count"
            " FROM conversations c ORDER BY c.updated_at DESC LIMIT ?",
            (limit,)).fetchall()

    def get_conversation(self, conversation_id: int) -> sqlite3.Row | None:
        """Get a conversation by id."""
        return self._conn.execute(
            "SELECT * FROM conversations WHERE id = ?",
            (conversation_id,)).fetchone()

    def last_conversation_id(self) -> int | None:
        """Get the id of the most recently updated conversation."""
        row = self._conn.execute(
            "SELECT id FROM conversations ORDER BY updated_at DESC LIMIT 1"
        ).fetchone()
        return row["id"] if row else None

    def close(self):
        self._conn.close()
