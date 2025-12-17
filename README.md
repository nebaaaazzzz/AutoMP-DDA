# MRDDA

Code and Dataset for "MRDDA: A Multi-Relational Graph Neural Network for Drug-Disease Association Prediction".

## Requirement

pythoh == 3.10.14

torch == 2.5.1

dgl >= 1.1.2

## Run

    python main.py

## GTN memory optimizations 🔧

If you enable the in-model GTN (set `use_gtn=True`), it can sometimes produce very large intermediate sparse matrices that cause GPU OOM. You can mitigate this by passing additional GTN options when creating the `Model`:

- `gtn_prune_max_edges` (int or None): keep only the top-k edges (by absolute weight) after sparse multiplications. Start with a small value (e.g., 5k–100k) and increase if results look noisy.
- `gtn_prune_eps` (float): drop edges with absolute weight <= this threshold (e.g., 1e-6).
- `gtn_run_on_cpu` (bool): if True, heavy sparse multiplications are performed on CPU to reduce GPU memory pressure (slower).

Example:

```python
model = Model(..., use_gtn=True, gtn_channels=2, gtn_layers=2,
              gtn_prune_max_edges=50000, gtn_prune_eps=1e-8, gtn_run_on_cpu=False)
```

Other options:

- Reduce `gtn_channels` or `gtn_layers` to reduce intermediate sizes.
- Precompute GTN embeddings offline or once per epoch and cache them instead of recomputing every batch.
- Call GTN in evaluation mode (`eval=True`), which now runs without building gradients and uses less memory.
