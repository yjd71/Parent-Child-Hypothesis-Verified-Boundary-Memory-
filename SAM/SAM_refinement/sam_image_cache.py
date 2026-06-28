"""Compatibility import for the SAM image embedding cache.

The implementation lives next to the SVB output cache so both cache policies
can be reviewed together without sharing an enable flag.
"""

try:
    from .svb_cache import SAMImageEmbeddingCache
except ImportError:
    from svb_cache import SAMImageEmbeddingCache


__all__ = ["SAMImageEmbeddingCache"]
