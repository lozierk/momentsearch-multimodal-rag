"""Put the repo root on sys.path so `import src.…` works with plain `pytest`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
