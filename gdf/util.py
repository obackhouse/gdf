"""Utility functions.
"""

import contextlib

import numpy as np

from gdf.kpts import KPoints


def cache(func):
    """Decorator to cache function results for class methods.

    Parameters
    ----------
    func : callable
        Function to cache.

    Returns
    -------
    wrapper : callable
        Wrapped function.
    """

    cache = {}

    def hashable(x):
        """Return a hashable version of `x`, handling k-point types."""
        if isinstance(x, KPoints):
            x = x._kpts
        elif hasattr(x, "kpts"):  # PySCF kpts container
            x = x.kpts
        if isinstance(x, np.ndarray):
            return KPoints._hash_kpts(x)
        else:
            return x

    def wrapper(self, *args, **kwargs):
        """Wrapped function."""
        key = tuple(hashable(arg) for arg in args)
        key += tuple((k, hashable(v)) for k, v in kwargs.items())
        if key not in cache:
            cache[key] = func(self, *args, **kwargs)
        return cache[key]

    return wrapper
