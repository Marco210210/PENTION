# script_test/conftest.py
import sys, pathlib

# project root = due livelli sopra questo file (…/PENTION)
ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
