"""C library functions.
"""

import ctypes
import os

import numpy as np

# Find the path to the library
path = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(path, "lib")

# Load the library
lib = np.ctypeslib.load_library("gdf.so", path)


def __getattr__(key):
    """Get a function from the library."""
    return getattr(lib, key)
