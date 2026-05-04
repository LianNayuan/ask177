"""Build/sync glossary from knowledge/glossary.md into the database.

Usage:
  python build_glossary.py              # sync file → DB (replaces all)
  python build_glossary.py --dry-run    # preview changes without writing
"""

import sys
from pathlib import Path

from database import Database

GLOSSARY_FILE = Path("knowledge/glossary.md")


def parse_glossary(path: Path) -> dict[str, str]:
    """Parse slang | formal pairs from a markdown file."""
    entries: dict[str, str] = {}
    if not path.is_file():
        print(f"File not found: {path}")
        return entries
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            slang, formal = line.split("|", 1)
            entries[slang.strip()] = formal.strip()
    return entries


def sync(db: Database, entries: dict[str, str]) -> int:
    """Replace all glossary entries in DB. Returns count written."""
    # Clear existing
    old = db.load_glossary()
    for slang in old:
        db.del_glossary(slang)
    # Write new
    for slang, formal in entries.items():
        db.add_glossary(slang, formal)
    return len(entries)


if __name__ == "__main__":
    dry_run = "--dry-run" in sys.argv

    incoming = parse_glossary(GLOSSARY_FILE)
    if not incoming:
        print("No entries found in glossary.md")
        sys.exit(0)

    db = Database()
    current = db.load_glossary()

    added = {k: v for k, v in incoming.items() if k not in current}
    removed = {k: v for k, v in current.items() if k not in incoming}
    changed = {k: v for k, v in incoming.items()
               if k in current and current[k] != v}

    print(f"file:  {len(incoming)} entries")
    print(f"db:    {len(current)} entries")
    if added:
        print(f"  + {len(added)} new: {dict(added)}")
    if removed:
        print(f"  - {len(removed)} removed: {dict(removed)}")
    if changed:
        print(f"  ~ {len(changed)} changed: {dict(changed)}")
    if not added and not removed and not changed:
        print("  (already in sync)")

    if dry_run:
        print("\n[dry-run] No changes written.")
    else:
        n = sync(db, incoming)
        print(f"\nSynced: {n} entries written to data.db")
