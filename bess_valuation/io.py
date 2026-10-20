"""Output folder handling: ``results/<name>/<YYYYMMDD_HHMMSS>/``."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

DEFAULT_RESULTS_ROOT = Path("results")


def new_run_dir(name: str, root: str | Path = DEFAULT_RESULTS_ROOT) -> Path:
    """Create a fresh timestamped run folder and return it.

    Existing folders are never reused: if the timestamp already exists (two runs in
    the same second), a numeric suffix is appended.

    Args:
        name: First-level folder, e.g. the battery config name or "price_generator".
        root: Results root directory.

    Returns:
        The created directory ``root/name/<YYYYMMDD_HHMMSS>[_k]``.
    """
    base = Path(root) / name
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = base / stamp
    k = 1
    while path.exists():
        path = base / f"{stamp}_{k}"
        k += 1
    path.mkdir(parents=True)
    return path
