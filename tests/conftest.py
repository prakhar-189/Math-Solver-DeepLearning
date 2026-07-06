import os
import sys

# Repo root on sys.path so tests can import utils / model / inference like the app does.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
