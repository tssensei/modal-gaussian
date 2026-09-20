# Historical neural strategies

These implementations remain for explicit old configurations and saved-result
validation/replay. Current v16 training lives in `motion.neural.component_field`.

| File | Historical behavior |
|---|---|
| `control_propagation.py` | CPU adaptive Dijkstra/process pool; explicit `--propagation-backend cpu` or numerical comparisons only |
| `fragment_propagation.py` | v9 post-training fragment attachment/transfer |
| `training_fragments.py` | v10 fragment transfer compiled into interpolation |
| `surface_attachments.py` | v11 surface-based attachment |
| `pointwise_attachments.py` | v12 per-point neighbor transfer |
| `guarded_attachments.py` | v14 stable donors and observable local residuals |
| `observation_refinement.py` | v13/v15 optional observation correction |
| `schema.py` | Lightweight persisted per-mode array names shared with loaders |

`motion.neural.strategies` owns dispatch. Subgraph, visibility and nearest-point
helpers live in `motion.common`. The historical modules no longer contain their
own copies. Shared GNN/field math remains in `motion.neural.neural_field` so old
network keys and baked fields retain their original meaning.

Existing CLI strategy switches and unified artifact loading remain available.
Python imports for these files must use `motion.legacy.neural`.
