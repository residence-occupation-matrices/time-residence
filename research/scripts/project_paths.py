"""Input and output path configuration for the study scripts.

``TIME_RESIDENCE_DATA_DIR`` and ``TIME_RESIDENCE_OUTPUT_DIR`` override the
repository's ``data`` and ``outputs`` directories.
"""

from __future__ import annotations

import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("TIME_RESIDENCE_DATA_DIR", REPOSITORY_ROOT / "data")).expanduser()
OUTPUT_ROOT = Path(
    os.environ.get("TIME_RESIDENCE_OUTPUT_DIR", REPOSITORY_ROOT / "outputs")
).expanduser()
EPIDEMIOLOGY_CSV = Path(
    os.environ.get(
        "TIME_RESIDENCE_EPIDEMIOLOGY_CSV",
        DATA_ROOT / "epidemiology" / "SISVER_Agebs_Zonas.csv",
    )
).expanduser()


def prepare_output(path: str | Path) -> Path:
    """Create the parent directory for an output and return its path."""

    result = Path(path)
    result.parent.mkdir(parents=True, exist_ok=True)
    return result
