import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from macromill_sentiment.cli import main

raise SystemExit(main(sys.argv[1:]))
