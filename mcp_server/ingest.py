"""
Ingestion pipeline for the Godot RAG agent.

Pipeline: .rst files -> parsed sections -> chunks (with metadata) -> embeddings -> Chroma
"""

import re
from dataclasses import dataclass, field
from pathlib import Path

from chromadb import PersistentClient
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from sentence_transformers import SentenceTransformer
from git import Repo

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / ".cache"
DOCS_CACHE_DIR = CACHE_DIR / "godot-docs"
CHROMA_PATH = CACHE_DIR / "chroma_db"
GODOT_DOCS_URL = "https://github.com/godotengine/godot-docs.git"

CODE_BLOCK_RE = re.compile(r"::\n\n((?:[ \t]+.*\n?)+)", re.MULTILINE)

# Matches a ".. tabs::" directive and captures everything indented deeper
# than it (the nested code-tab entries), stopping at the first line that
# dedents back to the tabs directive's own level.
TABS_BLOCK_RE = re.compile(
    r"^(?P<indent>[ \t]*)\.\.[ \t]+tabs::[ \t]*\n"
    r"(?P<body>(?:\n|(?P=indent)[ \t]+.*\n)*)",
    re.MULTILINE,
)

# Matches one ".. code-tab:: <lang> [label]" entry within a tabs block body
# and captures its code, stopping before the next code-tab entry.
CODE_TAB_RE = re.compile(
    r"^[ \t]*\.\.[ \t]+code-tab::[ \t]*(?P<lang>\S+)(?:[ \t]+(?P<label>.*?))?[ \t]*\n"
    r"(?P<code>(?:\n|(?![ \t]*\.\.[ \t]+code-tab::)[ \t]+.*\n)*)",
    re.MULTILINE,
)

MAX_CHUNK_CHARS = 1800  # rough proxy for ~400-500 tokens
CHROMA_BATCH_SIZE = 1000  # Chroma enforces a max batch size far smaller than a whole-docs ingest

HEADING_UNDERLINE_CHARS = "=-~^\"'"

@dataclass
class Chunk:
    """A chunk of text with associated metadata."""
    text: str
    source_path: str
    heading_path: str  # e.g. "Scripting > GDScript > Signals"
    chunk_type: str  # "prose" or "code"
    language: str = "any"  # "gdscript" | "cpp" | "csharp" | "any" (prose/unspecified)
    godot_version: str = "4.x"
    chunk_id: str = field(default="")


def sync_godot_docs() -> tuple[bool, str]:
    """
    Ensures the Godot docs repository is present and up to date.
    Returns (has_changes, current_commit_hash).
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not DOCS_CACHE_DIR.exists():
        print("Cloning Godot docs repository...")
        # Shallow clone (--depth 1) fetches only the latest commit to save disk and bandwidth
        repo = Repo.clone_from(GODOT_DOCS_URL, str(DOCS_CACHE_DIR), depth=1)
        return True, repo.head.commit.hexsha

    repo = Repo(str(DOCS_CACHE_DIR))
    old_commit = repo.head.commit.hexsha

    # Fetch latest remote changes
    origin = repo.remotes.origin
    origin.fetch()

    # Pull changes
    origin.pull()
    new_commit = repo.head.commit.hexsha

    has_changes = old_commit != new_commit
    return has_changes, new_commit


def split_into_sections(rst_text: str) -> list[tuple[str, str]]:
    """Return a list of (heading_path, section_text) tuples."""
    lines = rst_text.splitlines()
    sections: list[tuple[str, str]] = []
    heading_stack: list[str] = []
    current_lines: list[str] = []

    def flush():
        if current_lines:
            heading_path = " > ".join(heading_stack) if heading_stack else "(intro)"
            sections.append((heading_path, "\n".join(current_lines).strip()))

    i = 0
    while i < len(lines):
        line = lines[i]
        next_line = lines[i + 1] if i + 1 < len(lines) else ""

        is_heading = (
            next_line
            and len(next_line) >= len(line)
            and len(set(next_line.strip())) == 1
            and next_line.strip()[0] in HEADING_UNDERLINE_CHARS
        )

        if is_heading:
            flush()
            current_lines = []
            level = HEADING_UNDERLINE_CHARS.index(next_line.strip()[0])
            heading_stack = heading_stack[:level] + [line.strip()]
            i += 2  # skip the underline
            continue

        current_lines.append(line)
        i += 1

    flush()
    return sections

def chunk_section(heading_path: str, text: str, source_path: str) -> list[Chunk]:
    """Return a list of Chunks for a given section, splitting out any
    `.. tabs::` blocks into one Chunk per language."""
    chunks: list[Chunk] = []
    tabs_matches = list(TABS_BLOCK_RE.finditer(text))

    if not tabs_matches:
        return _chunk_plain_text(text, heading_path, source_path)

    last_end = 0
    for tmatch in tabs_matches:
        before = text[last_end:tmatch.start()]
        chunks.extend(_chunk_plain_text(before, heading_path, source_path))
        chunks.extend(_chunk_tabs_block(tmatch, heading_path, source_path))
        last_end = tmatch.end()

    trailing = text[last_end:]
    chunks.extend(_chunk_plain_text(trailing, heading_path, source_path))
    return chunks


def _chunk_plain_text(text: str, heading_path: str, source_path: str) -> list[Chunk]:
    """Original prose/code-block splitting logic (unchanged), now tagging language."""
    chunks: list[Chunk] = []
    code_blocks = list(CODE_BLOCK_RE.finditer(text))
    if not code_blocks:
        for piece in _split_prose(text, MAX_CHUNK_CHARS):
            chunks.append(Chunk(piece, source_path, heading_path, "prose", language="any"))
        return chunks

    last_end = 0
    for match in code_blocks:
        preceding = text[last_end:match.start()].strip()
        code = match.group(1).strip()
        if preceding:
            chunks.append(Chunk(preceding, source_path, heading_path, "prose", language="any"))
        if code:
            # Non-tabbed, single-language code blocks in these docs are
            # overwhelmingly GDScript — a reasonable default, not a guarantee.
            # Worth checking the where={"language": "gdscript"} distribution
            # after ingest to see how often this default is actually wrong.
            chunks.append(Chunk(code, source_path, heading_path, "code", language="gdscript"))
        last_end = match.end()

    trailing = text[last_end:].strip()
    if trailing:
        chunks.append(Chunk(trailing, source_path, heading_path, "prose", language="any"))
    return chunks


def _chunk_tabs_block(match: re.Match, heading_path: str, source_path: str) -> list[Chunk]:
    """Extract one Chunk per language variant from a `.. tabs::` block."""
    chunks: list[Chunk] = []
    for tab_match in CODE_TAB_RE.finditer(match.group("body")):
        lang = tab_match.group("lang").lower()  # first token only — "csharp", not "C#"
        code = _dedent(tab_match.group("code")).strip()
        if code:
            chunks.append(Chunk(code, source_path, heading_path, "code", language=lang))
    return chunks


def _dedent(text: str) -> str:
    """Strip the common leading whitespace from an indented RST block."""
    lines = text.splitlines()
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    if not indents:
        return text
    min_indent = min(indents)
    return "\n".join(l[min_indent:] if len(l) >= min_indent else l for l in lines)


def _split_prose(text: str, max_chars: int) -> list[str]:
    """Split a prose section into smaller pieces if it exceeds max_chars."""
    if len(text) <= max_chars:
        return [text] if text.strip() else []

    paragraphs = text.split("\n\n")
    pieces, current = [], ""
    for para in paragraphs:
        if len(current) + len(para) > max_chars and current:
            pieces.append(current.strip())
            current = para
        else:
            current += "\n\n" + para
    if current.strip():
        pieces.append(current.strip())
    return pieces


def add_in_batches(collection, ids: list[str], documents: list[str], metadatas: list[dict], batch_size: int = CHROMA_BATCH_SIZE):
    """Chroma rejects very large add() requests. Split large payloads into safe batches."""
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i + batch_size],
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
        )

def build_index(docs_dir: Path, collection_name: str = "godot_docs"):
    """Builds a Chroma index from the Godot docs."""
    model = SentenceTransformer("BAAI/bge-small-en-v1.5")  # local, free
    CHROMA_PATH.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(CHROMA_PATH))

    embedding_function = SentenceTransformerEmbeddingFunction(
        model_name="BAAI/bge-small-en-v1.5",
        normalize_embeddings=True,  # pairs with cosine space below
    )
    collection = client.get_or_create_collection(
        collection_name,
        embedding_function=embedding_function,
        metadata={"hnsw:space": "cosine"},  # cosine distance is more appropriate for normalized embeddings
    )
    all_chunks: list[Chunk] = []
    for rst_file in docs_dir.rglob("*.rst"):
        text = rst_file.read_text(encoding="utf-8", errors="ignore")
        for heading_path, section_text in split_into_sections(text):
            all_chunks.extend(chunk_section(heading_path, section_text, str(rst_file)))

    print(f"Built {len(all_chunks)} chunks from {docs_dir}")

    texts = [c.text for c in all_chunks]
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    metadatas = [
        {
            "source": c.source_path,
            "heading_path": c.heading_path,
            "chunk_type": c.chunk_type,
            "godot_version": c.godot_version,''
            'language': c.language,
        }
        for c in all_chunks
    ]

    add_in_batches(collection, ids, texts, metadatas, batch_size=CHROMA_BATCH_SIZE)
    print(f"Indexed {len(all_chunks)} chunks into '{collection_name}'")

def run_ingestion_pipeline():
    """Run the full ingestion pipeline: sync docs, build index if needed."""
    has_changes, commit_hash = sync_godot_docs()

    client = PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_or_create_collection("godot_docs")
    is_empty = collection.count() == 0

    if has_changes or is_empty:
        print(f"Rebuilding Godot docs index (changes={has_changes}, empty={is_empty}).")
        client.delete_collection("godot_docs")
        build_index(DOCS_CACHE_DIR)
    else:
        print(f"No changes in Godot docs (commit {commit_hash}); collection already populated.")

if __name__ == "__main__":
    run_ingestion_pipeline()