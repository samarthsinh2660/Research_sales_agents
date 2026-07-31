"""Makes `src.*` imports resolve regardless of pytest's invocation directory
(sales_outreach uses implicit namespace packages, no __init__.py files)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
