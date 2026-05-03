"""Crawl Splatoon weapon data from splatoonwiki.org (Inkipedia).

Usage:
  python crawl_en.py                          # crawl all Splatoon 3 weapons
  python crawl_en.py --game splatoon          # Splatoon 1 weapons
  python crawl_en.py --game splatoon2         # Splatoon 2 weapons
  python crawl_en.py --game all               # all games
  python crawl_en.py --page "Custom Splattershot Jr."  # single page
  python crawl_en.py --force                  # re-download existing files

Output: knowledge/wiki_en/<Weapon Name>.md
"""

import html as _html
import json
import re
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

API_BASE = "https://splatoonwiki.org/w/api.php"
OUTPUT_DIR = Path("knowledge/wiki_en")
DELAY = 1.0  # seconds between requests

GAME_CATEGORIES = {
    "splatoon3": "Category:Splatoon_3_main_weapons",
    "splatoon2": "Category:Splatoon_2_main_weapons",
    "splatoon":  "Category:Splatoon_main_weapons",
}

GAME_HEADINGS = {"Splatoon", "Splatoon 2", "Splatoon 3"}
SKIP_HEADINGS = {"contents", "gallery", "demonstration",
                 "references", "external links", "notes",
                 "data", "version history", "quotes", "badges",
                 "super smash bros. ultimate",
                 "weapon freshness rewards", "competitive",
                 "translation notes"}

# Image-caption-like patterns to filter from text
IMAGE_CAPTION_PATTERNS = [
    re.compile(p, re.IGNORECASE) for p in [
        r'^promotional image$',
        r'^2d icon$',
        r'^3d art(work)?',
        r'^splatnet icon$',
        r'^concept art',
        r'^titlecard',
        r'^weapon icon',
        r'^model of',
        r'^holding (the )?',
        r'^throwing ',
        r'^artwork of ',
        r'^another ',
        r'^three ',
        r'^the .* holding',
        r'^the .* used by',
        r'^held by ',
        r'^inkling (boy|girl) ',
        r'^a (team|rabbit|line) ',
        r'^two inklings',
        r'^sunken scroll',
        r'^the (same )?3d art',
        r'^the splattershot (on|being|as)',
        r'^the tableturf battle card',
        r'^a splattershot mem cake',
        r'^the (closest|center|second|third)',
        r'^the info page',
        r'^the \S+ on the',
        r'^\( link to file \)$',
        r'^link to file$',
    ]
]


# ── API helpers ─────────────────────────────────────────────────────

def api_request(params: dict) -> dict:
    """Make a MediaWiki API request. Returns parsed JSON."""
    params["format"] = "json"
    url = API_BASE + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={
        "User-Agent": "SplatoonRAG/1.0 (https://github.com/example; example@example.com)"
    })
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read())


# ── Weapon list ─────────────────────────────────────────────────────

def get_weapon_pages(category: str) -> list[str]:
    """Get all main weapon page titles from a category."""
    pages: list[str] = []
    cmcontinue = None
    while True:
        params = {
            "action": "query",
            "list": "categorymembers",
            "cmtitle": category,
            "cmlimit": 200,
        }
        if cmcontinue:
            params["cmcontinue"] = cmcontinue
        data = api_request(params)
        items = data.get("query", {}).get("categorymembers", [])
        for item in items:
            title = item["title"]
            if title.startswith("List of") or title.startswith("Category:"):
                continue
            pages.append(title)
        if "continue" in data:
            cmcontinue = data["continue"]["cmcontinue"]
        else:
            break
        time.sleep(0.3)
    return pages


# ── Page fetching ───────────────────────────────────────────────────

def fetch_weapon_html(page_title: str) -> str:
    """Fetch parsed HTML for a weapon page via the API."""
    params = {
        "action": "parse",
        "page": page_title,
        "prop": "text",
    }
    data = api_request(params)
    return data["parse"]["text"]["*"]


# ── HTML → Text helpers ────────────────────────────────────────────

def decode_entities(text: str) -> str:
    """Decode all HTML entities including numeric ones like &#91; → [."""
    return _html.unescape(text)


def strip_tags(text: str) -> str:
    """Remove HTML tags, decode all entities, collapse whitespace."""
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Remove citation superscripts
    text = re.sub(r'<sup[^>]*>.*?</sup>', '', text)
    # Remove all remaining HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    text = decode_entities(text)
    # Remove [edit] links (spaces from span separation)
    text = re.sub(r'\[\s*edit\s*\]', '', text)
    # Remove "--> Continue Dismiss <--" gallery artifacts
    text = re.sub(r'-->\s*Continue Dismiss\s*<!--', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def strip_citation_refs(text: str) -> str:
    """Remove standalone citation brackets like [1] [2] [a] [b]."""
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\[[a-z]\]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── Infobox parsing ─────────────────────────────────────────────────

def parse_game_infobox(html: str, game_heading: str) -> dict[str, str]:
    """Extract the weapon stat table for a specific game section."""
    # Find <h2> containing the game heading (may have <i> wrapper)
    pattern = re.compile(
        rf'<h2[^>]*>(.*?)</h2>',
        re.DOTALL
    )
    section_start = None
    for m in pattern.finditer(html):
        heading_text = re.sub(r'<[^>]+>', '', m.group(1)).strip()
        heading_text = re.sub(r'\[\s*edit\s*\]', '', heading_text).strip()
        if heading_text == game_heading:
            section_start = m.end()
            break

    if section_start is None:
        return {}

    # Find the next h2 to bound this section
    next_h2 = re.search(r'<h2[^>]*>', html[section_start:])
    if next_h2:
        section_end = section_start + next_h2.start()
    else:
        section_end = len(html)

    section_html = html[section_start:section_end]

    # Also check the page header (before first h2) for weapons whose
    # stats table is at the top of the page rather than inside an h2
    first_h2 = re.search(r'<h2[^>]*>', html)
    header_html = html[:first_h2.start()] if first_h2 else ""

    # Find all tables in both areas, pick the one with weapon stat labels
    all_tables = re.findall(r'<table[^>]*>(.*?)</table>', header_html + section_html, re.DOTALL)
    best_table = None
    best_rows: list[tuple[str, str]] = []

    weapon_labels = {"category", "class", "sub", "special", "range", "damage",
                     "fire rate", "charge speed", "mobility", "level"}

    for table_html in all_tables:
        rows = re.findall(
            r'<td[^>]*>\s*(.*?)\s*</td>\s*<td[^>]*>(.*?)</td>',
            table_html, re.DOTALL
        )
        # Check if this table has weapon stat labels
        label_texts = {strip_tags(l).lower() for l, v in rows}
        if label_texts & weapon_labels:
            best_table = table_html
            best_rows = rows
            break

    if not best_table:
        return {}

    fields: dict[str, str] = {}

    for label_html, value_html in best_rows:
        label = strip_tags(label_html)
        if not label or label.lower() in ("basic information",):
            continue

        # Try to extract stat value from text first (e.g. "52 / 100")
        text_value = strip_tags(value_html)
        stat_text_match = re.search(r'(\d+)\s*/\s*100', text_value)
        if stat_text_match:
            value = stat_text_match.group(1) + "/100"
        elif text_value:
            value = text_value
        else:
            continue

        if label == value:
            continue

        fields[label] = value

    return fields


# ── Text sections ───────────────────────────────────────────────────

def parse_text_sections(html: str) -> list[tuple[str, str]]:
    """Extract h2/h3 sections as (heading, content) pairs."""
    sections: list[tuple[str, str]] = []

    # Split by h2 headings
    h2_blocks = re.split(r'<h2[^>]*>\s*(.*?)\s*</h2>', html, flags=re.DOTALL)

    for i in range(1, len(h2_blocks) - 1, 2):
        heading = strip_tags(h2_blocks[i])
        content_html = h2_blocks[i + 1]

        heading_lower = heading.lower()

        # Skip ToC, gallery, references, game data sections
        if heading_lower in SKIP_HEADINGS:
            continue

        # Skip game-specific sections (handled by infobox)
        if heading in GAME_HEADINGS:
            continue

        # Extract paragraphs and list items
        # Get <p> content
        paragraphs = re.findall(r'<p>(.*?)</p>', content_html, re.DOTALL)
        # Get <li> content (from <ul> or <ol>)
        list_items = re.findall(r'<li>(.*?)</li>', content_html, re.DOTALL)

        clean_items: list[str] = []
        for p in paragraphs + list_items:
            text = strip_tags(p)
            text = strip_citation_refs(text)
            if not text:
                continue
            # Skip image-caption-only lines
            if any(pat.match(text) for pat in IMAGE_CAPTION_PATTERNS):
                continue
            clean_items.append(text)

        if clean_items:
            sections.append((heading, "\n\n".join(clean_items)))

    return sections


# ── Description ─────────────────────────────────────────────────────

def extract_description(html: str) -> str:
    """Extract the weapon description - the first <p> before the ToC."""
    # The description comes right after the infobox-gobbler, before <div id="toc"
    # Pattern: </div></div> <dl>...</dl> <p>DESC</p>
    # or just the first <p> with meaningful content
    toc_idx = html.find('<div id="toc"')
    if toc_idx < 0:
        toc_idx = len(html)

    prefix = html[:toc_idx]

    # Find the last <p> before ToC - that's the description
    paragraphs = re.findall(r'<p>(.*?)</p>', prefix, re.DOTALL)
    for p in reversed(paragraphs):
        text = strip_tags(p)
        if text and len(text) > 30:
            # Skip short dablinks
            if text.startswith("For other"):
                continue
            return text

    return ""


# ── Names in other languages ────────────────────────────────────────

def parse_names_in_other_languages(html: str) -> list[tuple[str, str]]:
    """Extract names in other languages from the table."""
    # Find the section heading
    section_match = re.search(
        r'<span[^>]*id="Names_in_other_languages"[^>]*>',
        html
    )
    if not section_match:
        return []

    # Find the heading end (could be h2 or h3)
    sidx = section_match.start()
    # Find which heading tag contains this span
    h_before = html.rfind('<h2', 0, sidx)
    h_before3 = html.rfind('<h3', 0, sidx)
    if h_before3 > h_before:
        heading_end = html.find('</h3>', sidx)
        if heading_end < 0:
            heading_end = html.find('</h2>', sidx)
    else:
        heading_end = html.find('</h2>', sidx)
        if heading_end < 0:
            heading_end = html.find('</h3>', sidx)
    if heading_end < 0:
        return []

    start = heading_end

    # Find the table (can be long - up to 10000 chars)
    table_match = re.search(r'<table[^>]*>(.*?)</table>', html[start:start + 12000], re.DOTALL)
    if not table_match:
        return []

    table_html = table_match.group(1)
    names: list[tuple[str, str]] = []

    # Extract table rows
    rows = re.findall(
        r'<tr[^>]*>(.*?)</tr>',
        table_html, re.DOTALL
    )

    last_lang = ""
    for row in rows:
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.DOTALL)
        if len(cells) >= 2:
            lang = strip_tags(cells[0]).rstrip(":")
            name = strip_tags(cells[1])
            name = strip_citation_refs(name)

            # Skip header rows
            if lang.lower() in ("language", "meaning", "notes", ""):
                if not lang:
                    # rowspan continuation: reuse last language
                    lang = last_lang
                else:
                    continue
            if not name:
                continue

            last_lang = lang
            names.append((lang, name))

    return names


# ── Markdown builder ────────────────────────────────────────────────

def build_markdown(title: str, html: str) -> str:
    """Convert a weapon page to markdown."""
    lines: list[str] = []
    lines.append(f"# {title}")
    lines.append("")

    # ── Description ────────────────────────────────────────────────
    desc = extract_description(html)
    if desc:
        lines.append(desc)
        lines.append("")

    # ── Splatoon 3 infobox ─────────────────────────────────────────
    for game in ("Splatoon 3", "Splatoon 2", "Splatoon"):
        infobox = parse_game_infobox(html, game)
        if not infobox:
            infobox = parse_game_infobox(html, f"<i>{game}</i>")
        if infobox:
            lines.append(f"## {game}")
            lines.append("")
            for label, value in infobox.items():
                if label.lower() in ("category",):
                    continue
                if not value or value == label:
                    continue
                lines.append(f"- **{label}**: {value}")
            lines.append("")

    # ── Text sections ──────────────────────────────────────────────
    sections = parse_text_sections(html)
    for heading, content in sections:
        lines.append(f"## {heading}")
        lines.append("")
        lines.append(content)
        lines.append("")

    # ── Names in other languages ───────────────────────────────────
    names = parse_names_in_other_languages(html)
    if names:
        lines.append("## Names in Other Languages")
        lines.append("")
        for lang, name in names:
            lines.append(f"- **{lang}**: {name}")
        lines.append("")

    return "\n".join(lines)


# ── File saving ─────────────────────────────────────────────────────

def sanitize_filename(name: str) -> str:
    """Sanitize weapon name for use as filename."""
    name = re.sub(r'[<>:"/\\|?*]', "", name)
    return name.strip()


def save_weapon(title: str, md: str) -> Path:
    """Save weapon markdown to wiki_en/."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fname = sanitize_filename(title) + ".md"
    path = OUTPUT_DIR / fname
    path.write_text(md, encoding="utf-8")
    return path


# ── Main ────────────────────────────────────────────────────────────

def crawl_page(page_title: str, force: bool = False) -> bool:
    """Crawl a single weapon page. Returns True on success."""
    fname = sanitize_filename(page_title) + ".md"
    out_path = OUTPUT_DIR / fname
    if out_path.exists() and not force:
        print(f"  SKIP (exists): {page_title}")
        return True

    try:
        html = fetch_weapon_html(page_title)
    except Exception as e:
        print(f"  FAIL fetch: {page_title} — {e}")
        return False

    md = build_markdown(page_title, html)
    save_weapon(page_title, md)
    print(f"  OK: {page_title} → {out_path}")
    return True


def main():
    force = "--force" in sys.argv
    game = "splatoon3"

    for i, a in enumerate(sys.argv[1:]):
        if a == "--game" and i + 2 < len(sys.argv):
            game = sys.argv[i + 2]
        elif a == "--page" and i + 2 < len(sys.argv):
            page = sys.argv[i + 2]
            print(f"Crawling single page: {page}")
            crawl_page(page, force=True)
            return

    if game == "all":
        categories = list(GAME_CATEGORIES.values())
    elif game in GAME_CATEGORIES:
        categories = [GAME_CATEGORIES[game]]
    else:
        print(f"Unknown game: {game}. Options: {list(GAME_CATEGORIES)} | all")
        sys.exit(1)

    all_pages: list[str] = []
    for cat in categories:
        pages = get_weapon_pages(cat)
        print(f"Found {len(pages)} weapons in {cat}")
        all_pages.extend(pages)

    seen: set[str] = set()
    unique: list[str] = []
    for p in all_pages:
        if p not in seen:
            seen.add(p)
            unique.append(p)
    all_pages = unique

    print(f"Total: {len(all_pages)} unique weapons to crawl")

    ok = fail = skip = 0
    for i, page in enumerate(all_pages):
        print(f"[{i+1}/{len(all_pages)}] {page}")
        fname = sanitize_filename(page) + ".md"
        if (OUTPUT_DIR / fname).exists() and not force:
            print(f"  SKIP (exists)")
            skip += 1
            continue

        if crawl_page(page, force=force):
            ok += 1
        else:
            fail += 1

        if i < len(all_pages) - 1:
            time.sleep(DELAY)

    print(f"\nDone: {ok} OK, {skip} skipped, {fail} failed")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
