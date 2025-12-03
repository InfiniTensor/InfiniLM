"""
InfiniLM C++ extension module
"""

import sys
import os
from pathlib import Path

# Ensure the directory containing this __init__.py is on sys.path
# This allows importing the .so file from the same directory
_lib_dir = Path(__file__).parent
if str(_lib_dir) not in sys.path:
    sys.path.insert(0, str(_lib_dir))

# Import the compiled C++ module
# The .so file should be installed in this directory by xmake
import _infinilm_llama

__all__ = ["_infinilm_llama"]
