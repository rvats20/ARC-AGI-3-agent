# ARC-AGI-3 Agent — Development Log & Findings

## Where we are

Working agent at `agent/my_agent.py` (v7: budget-aware frontier explorer).
Pipeline is fully operational: local play against all 25 live games,
`make submit` pushes to Kaggle, baseline v1 (random) already submitted.

## Hard-won engine facts (from live probing + reading obfuscated game code)

1. **Frame format**: `[1][64][64]` — each cell is a 64-vector (one-hot palette
   row). `_grid()` normalizes to 64×64 ints via first-nonzero index.
2. **Player tracking**: the player sprite's COLOR CHANGES across paint zones,
   so color-based tracking fails. Diff-centroid chaining works (moves ≤5px/step).
3. **Step budgets are HARD per level** (ls20: 42 actions). Exceeding budget →
   frame collapses to background, state stays NOT_FINISHED, only RESET restores.
4. **Lives (3) deplete on hazards**; 0 lives → level auto-resets with full budget.
5. **Game variety**: ls20 = movement maze; m0r0 = ACTION6 grid-click game with
   its own 150-action cap; available_actions differ per game.
6. ACTION6 requires `{"x": y, "y": y}` data dict or the engine throws KeyError.
7. RESET itself counts against MAX_ACTIONS (350 total in our config).

## Why levels=0 everywhere (honest assessment)

Level-1 completion requires solving the actual puzzle within ~42 actions:
ls20 needs collecting items in a maze where wrong moves cost lives AND steps.
A heuristic explorer can't do this reliably yet — it needs either:
- many lives of accumulated mapping + a near-optimal exploit path, or
- semantic understanding of each game's goal (what to collect/avoid).

The v7 agent builds exactly the cross-life memory infrastructure for the
first approach (persistent visited map, death blacklist, direction model,
best-path tracking), but 350 total actions only buys ~8 lives — not enough
to map a 64×64 maze and then solve it.

## Next steps (ranked)

1. Raise MAX_ACTIONS toward the 9h Kaggle limit (thousands of actions OK —
   the API is fast, 1000+ fps locally).
2. Per-game strategy fork keyed on `available_actions` + frame signatures.
3. Object permanence model: track ALL sprites (not just player) across
   frames to identify collectibles vs hazards vs walls from behavior.
4. Consider an LLM-in-the-loop variant for goal inference (public samples
   use VLM descriptions).

## Files

- `agent/my_agent.py` — current agent (edit only this)
- `scripts/play_local.py` — local harness (`--game X --max-steps N`)
- `scripts/debug_frames.py` — frame inspection helper
- `notebooks/kernel-metadata.json` — already set to rahulvats20
