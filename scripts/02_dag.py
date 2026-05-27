"""Render the DAG used for identification."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.dag import render_dag


def main() -> None:
    out = render_dag(ROOT / "figures" / "02_dag.png")
    print(f"DAG saved to {out}")


if __name__ == "__main__":
    main()
