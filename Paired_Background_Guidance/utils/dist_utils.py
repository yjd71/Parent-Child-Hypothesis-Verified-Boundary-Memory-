"""
Distributed utility helpers for prototype-guided training.

Planned contents:
- Gather per-rank prototypes before writing into the global bank.
- Keep bank states consistent across processes.
- Provide no-op fallbacks for single-GPU and non-distributed runs.
"""
