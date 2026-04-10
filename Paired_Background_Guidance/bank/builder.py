"""
Labeled-data prototype building logic.

Planned contents:
- Resize ground truth masks to decoder `p3` size.
- Build foreground/background masks.
- Perform per-image masked average pooling on `p3`.
- Append one foreground and one background prototype per valid labeled image.
- Optional distributed gather hooks before writing into the global bank.
"""
