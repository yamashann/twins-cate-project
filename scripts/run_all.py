"""Run the full pipeline end-to-end.

Executes, in order, every script under scripts/ that matches NN_*.py.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SCRIPTS_DIR = ROOT / "scripts"


def _load_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(seed: int = 0) -> None:
    scripts = sorted(p for p in SCRIPTS_DIR.glob("*.py") if p.name[:2].isdigit())
    for path in scripts:
        print(f"\n=== {path.name} ===")
        mod = _load_module(path)
        if hasattr(mod, "main"):
            try:
                mod.main(seed=seed)
            except TypeError:
                mod.main()


if __name__ == "__main__":
    main()
