"""Makes the repository's merian.py importable, unless merian is installed."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))
