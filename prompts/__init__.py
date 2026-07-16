from pathlib import Path

_DIR = Path(__file__).parent


def available_batches() -> list[str]:
    """Batch names are the .txt file stems in this folder."""
    return sorted(p.stem for p in _DIR.glob("*.txt"))


def load_batch(name: str) -> list[str]:
    path = _DIR / f"{name}.txt"
    if not path.exists():
        raise ValueError(f"Unknown batch '{name}'. Available: {available_batches()}")
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
