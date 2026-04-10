"""
Single coordination entry for Paired_Background_Guidance.

Planned responsibilities:
- Own and reset the dynamic foreground/background prototype banks.
- Dispatch labeled `p3 + gt` to bank building logic.
- Dispatch `p3` to retrieval and interaction logic to produce `H(x)`.
- Manage the learnable global scalar `mu`.
- Coordinate checkpoint save/load for prototype bank state and `mu`.
- Provide a clean interface to solver, evaluator, and inference code.
"""
