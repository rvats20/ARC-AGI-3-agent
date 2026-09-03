"""Probe a game's first frame to learn its available_actions, frame format,
initial grid, and per-cell palette. Used to design per-game strategies.
"""
from __future__ import annotations
import argparse
import json
import logging
import sys
from pathlib import Path
from collections import Counter

logging.disable(logging.CRITICAL)
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vendor" / "ARC-AGI-3-Agents"))

import numpy as np
import arc_agi
from arc_agi import OperationMode


def normalize_grid(frame):
    """Match the agent's _grid() helper: [1][64][64] -> 64x64 ints."""
    g = frame
    try:
        g = frame.tolist() if hasattr(frame, "tolist") else frame
    except Exception:
        return None
    if isinstance(g, (list, tuple)) and len(g) == 1:
        g = g[0]
    if g is None or len(g) == 0:
        return None
    first = g[0]
    if isinstance(first, (list, tuple, np.ndarray)) and len(first) \
            and isinstance(first[0], (list, tuple, np.ndarray)):
        return [[int(np.nonzero(np.asarray(c).flatten())[0][0])
                 if np.any(np.asarray(c)) else 0 for c in row] for row in g]
    try:
        return [[int(v) for v in row] for row in g]
    except Exception:
        return None


def probe_game(arc, game_id, n_steps=8):
    """Return a dict of structural signals about the game."""
    env = arc.make(game_id)
    f0_raw = env.reset()
    g0 = normalize_grid(f0_raw.frame)
    out = {
        "game_id": game_id.split("-")[0],
        "state": str(f0_raw.state),
        "available_actions": list(getattr(f0_raw, "available_actions", None) or []),
        "levels_completed": int(getattr(f0_raw, "levels_completed", 0)),
        "initial_grid": g0,
        "initial_palette": sorted({v for row in g0 for v in row}) if g0 else [],
    }
    # Probe first n steps with each action to see what changes
    from arcengine import GameAction
    avail = out["available_actions"] or [1, 2, 3, 4]
    # Coerce each int to its corresponding GameAction (covers available_actions
    # being ints, enums, or strings).
    avail_enums = []
    for a in avail:
        if isinstance(a, GameAction):
            avail_enums.append(a)
        elif isinstance(a, int):
            try:
                avail_enums.append(GameAction(a))
            except ValueError:
                pass
        elif isinstance(a, str):
            try:
                avail_enums.append(GameAction[a])
            except KeyError:
                pass
    if not avail_enums:
        avail_enums = [GameAction.ACTION1, GameAction.ACTION2,
                        GameAction.ACTION3, GameAction.ACTION4]
    action_effects = {}
    for ga in avail_enums:
        env.reset()
        env.step(ga)
        f1 = env.observation_space  # FrameDataRaw
        g1 = normalize_grid(f1.frame)
        if g0 and g1:
            diff = [(r, c) for r in range(64) for c in range(64) if g0[r][c] != g1[r][c]]
            action_effects[ga.name] = {
                "diff_cells": len(diff),
                "sample_diff": diff[:5],
                "new_state": str(f1.state),
            }
    out["action_effects"] = action_effects
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="analysis/game_probes.json")
    ap.add_argument("--games", nargs="*", default=None,
                    help="game ids to probe; default = all")
    args = ap.parse_args()

    arc = arc_agi.Arcade(operation_mode=OperationMode.NORMAL)
    all_envs = arc.get_environments()
    target_ids = [e.game_id for e in all_envs]
    if args.games:
        wanted = set(args.games)
        target_ids = [g for g in target_ids if g.split("-")[0] in wanted]
    out = []
    for gid in target_ids:
        try:
            info = probe_game(arc, gid)
            # Don't serialize the full grid in the summary, just key bits
            summary = {k: v for k, v in info.items() if k != "initial_grid"}
            summary["initial_grid_size"] = (len(info["initial_grid"]),
                                              len(info["initial_grid"][0]) if info["initial_grid"] else 0)
            summary["initial_grid_palette_size"] = len(info["initial_palette"])
            out.append(summary)
            print(f"  {gid}: palette_size={summary['initial_grid_palette_size']}, "
                  f"avail={summary['available_actions']}, "
                  f"action_effects={ {a: info['action_effects'][a]['diff_cells'] for a in info['action_effects']} }")
        except Exception as e:
            print(f"  {gid}: ERROR {e}")
    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {len(out)} game probes to {args.out}")


if __name__ == "__main__":
    main()
