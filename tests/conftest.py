import os
import sys

# Make app.py importable when pytest is run from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
