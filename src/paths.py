"""
Global project path file
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]   # src/paths.py -> repo root
DATA_DIR = PROJECT_ROOT / "data"
UNIVERSE_DIR = DATA_DIR / "production" / "universe"
UNIVERSE_RAW_DIR = UNIVERSE_DIR / "raw"