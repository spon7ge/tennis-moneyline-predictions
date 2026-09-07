# Final whole-branch review fixes

## Completed

- CI now runs the full suite with `uv run pytest`.
- B2 uses constant-zero imputation for antisymmetric difference features; a NaN swap regression verifies complementary probabilities.
- Tournament Elo updates both the match surface and an independent `Overall` rating using the same frozen pre-tournament state and level-specific K-factor.
- Each yearly supervised fit derives B0 and B2 temperatures from prequential OOF scores on the rolling training window. Sparse or otherwise unusable OOF windows explicitly fall back to neutral `T=1`.
- No-history form is represented as missing (`NaN`) rather than a fabricated `0.0`.
- Documentation now marks B2b/B3 and ranking snapshots as unimplemented, marks Sapling/dirty metadata as planned, and records that a full multi-year corpus run remains deferred.
- The missing-key odds test now explicitly overrides local `.env` files, allowing the full suite to remain deterministic.

## Verification

- `uv run pytest -q`
- Result: `101 passed`

## Known deferrals

- Full multi-year corpus execution was intentionally not run.
- B2b/B3, ranking snapshot ingestion/conflict resolution, and Sapling/dirty metadata capture remain outside the shipped v1 implementation.
