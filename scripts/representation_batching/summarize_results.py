#!/usr/bin/env python3
"""Build the portable HTML/CSV experiment ledger without loading a model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from representation_batching.report import main

if __name__ == "__main__":
    main()
