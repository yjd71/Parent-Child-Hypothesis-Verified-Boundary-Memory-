"""
Labeled-only supervision path for the learnable scalar `mu`.

Planned contents:
- Build the rectified labeled prediction `y_hat_r`.
- Compute supervision against labeled ground truth.
- Restrict gradients so that this loss updates `mu` only.
"""
