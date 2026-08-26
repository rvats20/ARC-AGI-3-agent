"""ARC-AGI-3 agent v7 — budget-aware frontier explorer (final prototype).

Consolidated engine knowledge:
- Step budgets are HARD per level (~42 in ls20); exceeding them collapses the
  frame to background but state stays NOT_FINISHED. Only RESET restores.
- Lives (3) deplete on hazards; at 0 → lose() → level resets automatically
  with full budget. Frame collapse also happens here.
- Player color mutates; track via diff-centroid chaining.
- Click games exist (m0r0: ACTION6 = x,y click; 150-action hard cap).

v7 policy:
1. On collapse detection: RESET immediately (only reliable way to continue).
2. Persistent across lives: world map, death blacklist, direction model,
   best-known path from spawn.
3. Each life: follow best_path prefix, then explore ONE new branch toward
   the nearest unvisited coarse cell; keep path length within observed
   budget minus safety margin.
4. Track budget: if steps_this_life approaches the smallest observed death
   step, start heading back / wrap up life cleanly.
"""
from __future__ import annotations

import random
import time
from collections import Counter, deque
from typing import Any

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


def _grid(frame):
    import numpy as np
    if frame is None:
        return None
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


def diff_cells(a, b):
    if a is None or b is None:
        return []
    h = min(len(a), len(b)); w = min(len(a[0]), len(b[0]))
    return [(r, c) for r in range(h) for c in range(w) if a[r][c] != b[r][c]]


def centroid(cells):
    if not cells:
        return None
    n = len(cells)
    return (sum(y for y, _ in cells)//n, sum(x for _, x in cells)//n)


class MyAgent(Agent):
    MAX_ACTIONS = 2000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        random.seed(hash(self.game_id) % 9999 + int(time.time()) % 10_000)
        # persistent across lives
        self.world_visited: set = set()
        self.death_cells: set = set()
        self.dir_map: dict[int, tuple[int, int]] = {}
        self.best_path: list = []
        self.best_depth = -1
        self.lives = 0
        self.min_death_step: int | None = None
        self.new_life()

    def new_life(self):
        self.prev_grid = None
        self.prev_action = None
        self.pos = None
        self.hist: deque = deque(maxlen=64)
        self.stagnation = 0
        self.steps = 0
        self.path: list = []
        self.visited_life: set = set()
        self.probe_left = [1, 2, 3, 4]
        self.last_click = None
        self.hot_cell = None
        self.hot_delta = -1
        self.click_idx = 0
        self.click_cells = [(x, y) for y in (10, 22, 32, 42, 54)
                            for x in (10, 22, 32, 42, 54)]
        random.shuffle(self.click_cells)

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}"

    def is_done(self, frames, latest_frame) -> bool:
        return latest_frame.state is GameState.WIN

    def _dead(self, grid) -> bool:
        """True when the frame collapses to a single flat color (level over).
        Note: dominance fails because the bg color itself changes; use
        unique-color count instead."""
        if grid is None:
            return False
        return len({v for row in grid for v in row}) <= 2

    def _track(self, grid):
        """Position = centroid of changed cells. The player is the main mover;
        always accept the diff centroid (jump-gate: pos may be None or stale
        from a previous life, so don't gate on proximity)."""
        c = centroid(diff_cells(self.prev_grid, grid))
        if c is not None:
            self.pos = c
        return self.pos

    def choose_action(self, frames, latest_frame) -> GameAction:
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._close_life()
            self.new_life()
            return GameAction.RESET

        grid = _grid(latest_frame.frame)

        if self.prev_grid is not None and not self._dead(self.prev_grid) \
                and self._dead(grid):
            self.lives += 1
            if self.steps > 3:
                if self.min_death_step is None or self.steps < self.min_death_step:
                    self.min_death_step = self.steps
                if len(self.visited_life | self.world_visited) > self.best_depth \
                        and len(self.path) >= 3:
                    self.best_depth = len(self.visited_life | self.world_visited)
                    self.best_path = list(self.path[:-1])  # drop fatal action
            self.new_life()
            return GameAction.RESET

        pos = self._track(grid) or (32, 32)
        key = (int(pos[0])//6*6, int(pos[1])//6*6)

        moved = bool(self.hist) and self.hist[-1] != key
        self.stagnation = 0 if moved else self.stagnation + 1
        self.hist.append(key)
        self.visited_life.add(key)
        self.world_visited.add(key)
        self.steps += 1

        avail = list(getattr(latest_frame, "available_actions", None) or [1,2,3,4])

        if self.prev_action in (1,2,3,4) and moved \
                and getattr(self, "_prev_key", None) and self._prev_key != key:
            dy = key[0]-self._prev_key[0]; dx = key[1]-self._prev_key[1]
            if max(abs(dy), abs(dx)) <= 12:
                self.dir_map[self.prev_action] = (dy, dx)
        self._prev_key = key

        # ── click games ──
        if 6 in avail:
            d = diff_cells(self.prev_grid, grid)
            if len(d) > self.hot_delta and self.last_click:
                self.hot_cell = self.last_click; self.hot_delta = len(d)
            a = GameAction.ACTION6
            if self.hot_cell is not None and self.click_idx >= len(self.click_cells):
                x = max(0, min(63, self.hot_cell[0]+random.randint(-5,5)))
                y = max(0, min(63, self.hot_cell[1]+random.randint(-5,5)))
            elif self.click_idx < len(self.click_cells):
                x, y = self.click_cells[self.click_idx]; self.click_idx += 1
            else:
                self.click_cells = [(random.randint(4,60), random.randint(4,60))
                                    for _ in range(25)]
                self.click_idx = 1
                x, y = self.click_cells[0]
            a.set_data({"x": x, "y": y})
            self.last_click = (x, y)
            self.prev_action = 6
            self.prev_grid = grid
            return a

        # ── movement games ──
        movable = [m for m in (1, 2, 3, 4) if m in avail] or [1,2,3,4]
        safe = [m for m in movable if (key, m) not in self.death_cells] or movable

        act = self._pick(safe, key)
        a = getattr(GameAction, f"ACTION{act}")
        a.reasoning = {"why": "frontier", "life": self.lives, "steps": self.steps}
        self.current_act = act
        self.prev_action = act
        self.path.append(act)
        self.prev_grid = grid
        return a

    def _close_life(self):
        self.lives += 1
        score = len(self.visited_life | self.world_visited)
        if score > self.best_depth and len(self.path) >= 2:
            self.best_depth = score
            self.best_path = list(self.path)

    def _pick(self, candidates, key) -> int:
        known = {a: d for a, d in self.dir_map.items() if a in candidates}
        best, best_score = None, None
        for act, d in known.items():
            t = (max(0, min(63, key[0]+d[0]*5)), max(0, min(63, key[1]+d[1]*5)))
            tk = (t[0]//6*6, t[1]//6*6)
            s = 0 if tk not in self.world_visited else 40
            if (tk, act) in self.death_cells:
                s += 200
            if best_score is None or s < best_score:
                best_score, best = s, act
        return best if best is not None else random.choice(candidates)
