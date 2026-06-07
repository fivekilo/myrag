from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = BACKEND_DIR.parent

TEMP_DIR = BACKEND_DIR / "temp"
SOURCE_DOCS_DIR = BACKEND_DIR / "00-source-docs"
LOADED_DOCS_DIR = BACKEND_DIR / "01-loaded-docs"
CHUNKED_DOCS_DIR = BACKEND_DIR / "01-chunked-docs"
EMBEDDED_DOCS_DIR = BACKEND_DIR / "02-embedded-docs"
VECTOR_STORE_DIR = BACKEND_DIR / "03-vector-store"
CHROMADB_DIR = VECTOR_STORE_DIR / "chromadb"
SEARCH_RESULTS_DIR = BACKEND_DIR / "04-search-results"
GENERATION_RESULTS_DIR = BACKEND_DIR / "05-generation-results"

RUNTIME_DIRS = [
    TEMP_DIR,
    SOURCE_DOCS_DIR,
    LOADED_DOCS_DIR,
    CHUNKED_DOCS_DIR,
    EMBEDDED_DOCS_DIR,
    VECTOR_STORE_DIR,
    SEARCH_RESULTS_DIR,
    GENERATION_RESULTS_DIR,
]


def ensure_runtime_dirs() -> None:
    for directory in RUNTIME_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


def workspace_relative(path: Path | str) -> str:
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(WORKSPACE_DIR).as_posix()
    except ValueError:
        return resolved.as_posix()
