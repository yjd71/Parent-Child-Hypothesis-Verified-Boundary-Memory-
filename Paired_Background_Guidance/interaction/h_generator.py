"""
Generate `H(x)` from low-resolution prototype interaction outputs.

Planned contents:
- Upsample `M(x)` to the original image resolution.
- Keep a consistent probability-map convention for downstream losses and fusion.
- Centralize resize policy for training, inference, and evaluation.
"""
