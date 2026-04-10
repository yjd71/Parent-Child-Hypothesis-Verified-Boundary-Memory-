# Paired_Background_Guidance

This package hosts the prototype-guided branch for the semi-supervised COD baseline.

## Module Layout

```text
Paired_Background_Guidance/
├─ __init__.py
├─ README.md
├─ structures.py
├─ manager.py
├─ bank/
├─ interaction/
├─ fusion/
├─ losses/
└─ utils/
```

## Responsibilities

- `structures.py`: shared data containers and interface contracts across the prototype branch.
- `manager.py`: the only coordination entry for training, inference, evaluator, and checkpoint workflow.
- `bank/`: dynamic prototype bank storage, labeled-feature collection, and retrieval preparation.
- `interaction/`: `Sim -> fu -> M(x) -> H(x)` interaction chain.
- `fusion/`: learnable `mu` parameter and rectified prediction formulas.
- `losses/`: probability-space supervision for `H(x)`, rectified outputs, and `mu`.
- `utils/`: tensor, mask, KDE, distributed, and checkpoint helper utilities.

## Planned Integration

- `talnet.py`: expose `p3`, `main_logit`, and image size to this package.
- `solver.py`: use `manager.py` to build banks, create `H(x)`, and manage `mu` updates.
- `evaluator.py`: reuse the same manager to generate rectified predictions at test time.
