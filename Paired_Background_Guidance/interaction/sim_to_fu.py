"""
Convert similarity matrices into fused retrieval responses.

Planned contents:
- Row-wise Top-k extraction for `Sim_fg` and `Sim_bg`.
- Temperature-scaled softmax attention over Top-k similarities.
- Weighted fusion to obtain `fu_fg` and `fu_bg`.
- Restore fused outputs back to spatial map format.
"""
