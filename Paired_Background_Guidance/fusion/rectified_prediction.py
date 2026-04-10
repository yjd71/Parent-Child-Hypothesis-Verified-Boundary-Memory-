"""
Rectified prediction formulas for labeled and unlabeled branches.

Planned contents:
- Labeled fusion: `y_hat_r = y_hat + (1 - mu) * H(x)`.
- Unlabeled fusion: `y_bar_r = y_bar + (1 - mu) * H(x)`.
- Probability clamping and numerical safety handling.
- Shared utilities for train-time and eval-time rectified outputs.
"""
