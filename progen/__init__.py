"""Compatibility package exposing the bundled ProGen2 modules.

The original ProGen2 files live under `progen2_AbELA/progen/` so the training
scripts can still run directly from that directory. This package lets users
import the same modules from the repository root, for example
`from progen.abela import load_abela_records`.
"""

from pathlib import Path

__path__ = [str(Path(__file__).resolve().parents[1] / "progen2_AbELA" / "progen")]
