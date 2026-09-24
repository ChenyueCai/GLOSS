# scripts/completion

Two-stage texture completion. Run both stages with `bash scripts/run_completion.sh <mesh> <view id>`; see [docs/completion.md](../../docs/completion.md) for inputs, outputs, and options.

- `complete_stage_1.py`: SyncMVD-style guidance texture.
- `complete_stage_2.py`: final texture from that guidance.
