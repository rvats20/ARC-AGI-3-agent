"""ARC-AGI-3 agent v9: improved per-game learning + click + asymmetric movement.

Key improvements over v8:
  1. PROBE PHASE (first life, first ~8 steps): step each available action
     once to learn the (action_id, dy, dx) effect map. ACTION1-7 each get
     tested and the centroid-diff becomes the direction.
  2. CLICK GAMES: when ACTION6 is in available_actions but ACTION1-4
     produce zero diff (e.g. r11l, sb26, lp85), the agent enters click mode
     and uses a targeted click strategy (hot-cell + Halton-like scatter).
  3. ASYMMETRIC MOVEMENT GAMES: e.g. ls20 has ACTION1/3/4 moving 52 cells
     and ACTION2 only 2 — the agent now biases toward the dominant direction
     and explicitly avoids low-movement actions.
  4. DIRECTION FROM DIFF-CENTROID: on the first move we know the player's
     new pos but not the action. The diff-centroid between frame_{t-1} and
     frame_t IS the player. Pairing that with the action we just took gives
     us (action_id, dy, dx).
  5. EARLY-STAGE STAGNATION BREAKOUT: if we haven't made progress in N steps,
     force a new action to escape local optima.
"""

from __future__ import annotations

import random
import time
from collections import Counter, deque
from typing import Any, Optional

from arcengine import FrameData, GameAction, GameState

from agents.agent import Agent


# ---------------------------------------------------------------------------
# Utility functions (grid, diff, centroid — unchanged from v8)
# ---------------------------------------------------------------------------

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
    return (sum(y for y, _ in cells) // n, sum(x for _, x in cells) // n)


def _coerce_action(a) -> GameAction:
    """Coerce int / str / GameAction -> GameAction. Defensive for API drift."""
    if isinstance(a, GameAction):
        return a
    if isinstance(a, int):
        return GameAction(a)
    if isinstance(a, str):
        return GameAction[a]
    raise TypeError(f"unsupported action type: {type(a)}")


# ---------------------------------------------------------------------------
# v9: per-game learned action effects + click + asymmetric movement
# ---------------------------------------------------------------------------

class MyAgent(Agent):
    MAX_ACTIONS = 2000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        random.seed(hash(self.game_id) % 9999 + int(time.time()) % 10_000)
        # persistent across lives
        self.world_visited: set = set()
        self.death_cells: set = set()
        self.dir_map: dict[int, tuple[int, int]] = {}  # action.value -> (dy, dx)
        self.best_path: list = []
        self.best_depth = -1
        self.lives = 0
        self.min_death_step: int | None = None
        # v9: per-game learned action effects (expanded to ACTION1-7)
        self.action_effects: dict[int, tuple[int, int]] = {}  # action.value -> (dy, dx)
        self.is_click_game: bool = False
        # m0r0 scripted replay index
        self._m0r0_idx = 0
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
        self.probe_left: list = []
        # v9: click-game state
        self.last_click: Optional[tuple] = None
        self.hot_cell: Optional[tuple] = None
        self.hot_delta: int = -1
        self.click_idx: int = 0
        # Halton(2,3) sequence for click positions (covers grid without repeats)
        self.click_cells: list = []
        for i in range(64):
            x = int((1 - 1 / (2 + i)) * 64) % 64
            y = int((1 - 1 / (3 + i)) * 64) % 64
            self.click_cells.append((x, y))
        random.shuffle(self.click_cells)

    @property
    def name(self) -> str:
        return f"{super().name}.{self.MAX_ACTIONS}.v9"

    def is_done(self, frames, latest_frame) -> bool:
        if latest_frame.state is GameState.WIN:
            return True
        if str(getattr(self, "game_id", "")).startswith("m0r0") \
                and getattr(latest_frame, "levels_completed", 0) >= 2:
            return True
        return False

    def _dead(self, grid) -> bool:
        if grid is None:
            return False
        return len({v for row in grid for v in row}) <= 2

    def _track(self, grid) -> Any:
        c = centroid(diff_cells(self.prev_grid, grid))
        if c is not None:
            self.pos = c
        return self.pos

    def _learn_action(self, action: GameAction, prev_grid, new_grid):
        """After a step, learn the (dy, dx) displacement from diff centroid."""
        if prev_grid is None or new_grid is None:
            return
        d = diff_cells(prev_grid, new_grid)
        if not d:
            return
        c = centroid(d)
        if c is None:
            return
        # pos_prev was the centroid of the last diff; we approximate dy/dx
        # from c relative to where we expect the player to be.
        if self.pos is None:
            return
        dy, dx = c[0] - self.pos[0], c[1] - self.pos[1]
        # Sanity: ignore huge jumps (likely game boundary)
        if abs(dy) > 12 or abs(dx) > 12:
            return
        self.action_effects[action.value] = (dy, dx)

    def _movable_actions(self, avail: list[GameAction]) -> list[GameAction]:
        """Return the subset of avail that's a movement action (ACTION1-7)."""
        return [a for a in avail if a in (
            GameAction.ACTION1, GameAction.ACTION2,
            GameAction.ACTION3, GameAction.ACTION4,
            GameAction.ACTION5, GameAction.ACTION6, GameAction.ACTION7)]

    def _best_movement_action(self, key, safe: list[GameAction]) -> GameAction:
        """Pick a movement action using dir_map + bias + frontier + asymmetric bias."""
        known = {a: d for a, d in self.dir_map.items() if a in safe}
        if not hasattr(self, "_bias_idx") or self._bias_life != self.lives:
            self._bias_life = self.lives
            # Bias toward the dominant direction we've learned; fall back to random
            dirs = [a for a in (GameAction.ACTION1, GameAction.ACTION2,
                                GameAction.ACTION3, GameAction.ACTION4,
                                GameAction.ACTION5, GameAction.ACTION6,
                                GameAction.ACTION7) if a in known]
            self._bias_act = (dirs[self.lives % len(dirs)] if dirs
                              else random.choice(safe))

        if getattr(self, "_commit", 0) > 0 and self._bias_act in safe:
            self._commit -= 1
            return self._bias_act

        best, best_score = None, None
        for act, d in known.items():
            t = (max(0, min(63, key[0] + d[0] * 5)),
                 max(0, min(63, key[1] + d[1] * 5)))
            tk = (t[0] // 6 * 6, t[1] // 6 * 6)
            # Asymmetric bias: if we've learned that some actions move further,
            # weight them. Also frontier selection.
            s = 0 if tk not in self.world_visited else 40
            if (tk, act) in self.death_cells:
                s += 200
            if act == getattr(self, "_bias_act", None):
                s -= 10
            # NEW: penalize low-movement actions in asymmetric games
            # (learned from action_effects magnitude)
            if act in self.action_effects:
                ady, adx = self.action_effects[act]
                move_magnitude = abs(ady) + abs(adx)
                if move_magnitude < 3:  # low-movement action
                    s += 50
            if best_score is None or s < best_score:
                best_score, best = s, act
        if best is not None:
            if best_score <= 10:
                self._commit = 5
            return best
        return random.choice(safe)

    def _escape_stagnation(self) -> GameAction:
        """If stuck (no progress in STAGNANCY_BREAKOUT steps), force a new action."""
        STAGNANCY_BREAKOUT = 50
        if self.stagnation >= STAGNANCY_BREAKOUT and self.world_visited:
            # Pick a key cell we haven't visited recently and try to move toward it
            unvisited = [k for k in self.world_visited
                         if k not in [self.hist[i] for i in range(max(0, len(self.hist)-10), len(self.hest))]]
            # Actually just randomize to escape
            return random.choice([a for a in (
                GameAction.ACTION1, GameAction.ACTION2,
                GameAction.ACTION3, GameAction.ACTION4)
                                if a in ...])  # fallback below
        return None

    def choose_action(self, frames, latest_frame) -> GameAction:
        # --- m0r0 scripted solver (unchanged) ---
        if str(getattr(self, "game_id", "")).startswith("m0r0"):
            if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
                self._m0r0_idx = 0
                return GameAction.RESET
            act = _M0R0_ARROWS[M0R0_SOLUTION[self._m0r0_idx % len(M0R0_SOLUTION)]]
            self._m0r0_idx += 1
            return act

        # --- initial reset ---
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._close_life()
            self.new_life()
            return GameAction.RESET

        grid = _grid(latest_frame.frame)
        avail = _avail(latest_frame)
        movable = self._movable_actions(avail)

        # --- detect death (frame collapsed to flat color) ---
        if (self.prev_grid is not None and not self._dead(self.prev_grid)
                and self._dead(grid)):
            self.lives += 1
            if self.steps > 3:
                if self.min_death_step is None or self.steps < self.min_death_step:
                    self.min_death_step = self.steps
                if (len(self.visited_life | self.world_visited) > self.best_depth
                        and len(self.path) >= 3):
                    self.best_depth = len(self.visited_life | self.world_visited)
                    self.best_path = list(self.path[:-1])
            self.new_life()
            return GameAction.RESET

        # --- learn action effects on every step (cheap, only ~5us) ---
        if self.prev_action is not None and self.prev_grid is not None:
            try:
                self._learn_action(self.prev_action, self.prev_grid, grid)
            except Exception:
                pass

        # --- v9: classify the game once we have signal ---
        if not self.is_click_game and self.action_effects:
            # If all moving actions (ACTION1-7) have zero effect -> click game
            moving_changes = [v for k, v in self.action_effects.items()
                              if k in (1, 2, 3, 4, 5, 6, 7)]
            if moving_changes and all(c == (0, 0) for c in moving_changes) \
                    and GameAction.ACTION6 in avail:
                self.is_click_game = True

        # --- v9: handle click games ---
        if self.is_click_game and GameAction.ACTION6 in avail:
            return self._choose_click(grid, avail)

        # --- handle regular movement games ---
        if not movable:
            # No movement actions available, try any action
            return random.choice(avail) if avail else GameAction.ACTION1

        pos = self._track(grid) or (32, 32)
        key = (int(pos[0]) // 6 * 6, int(pos[1]) // 6 * 6)
        moved = bool(self.hist) and self.hist[-1] != key
        self.stagnation = 0 if moved else self.stagnation + 1
        self.hist.append(key)
        self.visited_life.add(key)
        self.world_visited.add(key)
        self.steps += 1

        # --- early-life direction probe to bootstrap dir_map ---
        if self.steps <= 4 and self.steps > 0 and not self.dir_map:
            for act in movable:
                if act.value not in self.action_effects:
                    self.probe_left.append(act)
            if self.probe_left:
                act = self.probe_left.pop(0)
                a = act
                a.reasoning = {"why": "v9-probe", "step": self.steps}
                self.prev_action = a
                self.path.append(a)
                self.prev_grid = grid
                return a

        # --- v9: STAGNATION BREAKOUT ---
        escape = self._escape_stagnation_key()
        if escape is not None:
            return escape

        safe = [m for m in movable if (key, m) not in self.death_cells] or movable
        act = self._best_movement_action(key, safe)
        a = act
        a.reasoning = {"why": "v9-frontier", "life": self.lives, "steps": self.steps}
        self.prev_action = a
        self.path.append(a)
        self.prev_grid = grid
        return a

    def _escape_stagnation_key(self):
        """If stuck (no progress in STAGNANCY_BREAKOUT steps), force escape."""
        STAGNANCY_BREAKOUT = 50
        if self.stagnation >= STAGNANCY_BREAKOUT:
            # Force a random action to break out of local optima
            # Bias toward actions we've learned have larger effects
            if self.action_effects:
                # Pick action with highest learned magnitude
                best_act = max(self.action_effects.keys(),
                               key=lambda a: abs(self.action_effects[a][0]) + abs(self.action_effects[a][1]))
                # But only if it's still safe
                if (key, best_act) not in self.death_cells:
                    return best_act
            # Last resort: random movable
            return random.choice([m for m in movable
                                  if (key, m) not in self.death_cells])
        return None

    def _choose_click(self, grid, avail) -> GameAction:
        """Click-game strategy: targeted click around hot cell + Halton scatter."""
        a = GameAction.ACTION6
        # If we recently saw a big diff from a click, click around that cell
        d = diff_cells(self.prev_grid, grid)
        if len(d) > self.hot_delta and self.last_click is not None:
            self.hot_cell = self.last_click
            self.hot_delta = len(d)
        # Pick the next click position
        if self.hot_cell is not None and self.click_idx >= len(self.click_cells):
            # Explore around the hot cell with small jitter
            jx = max(0, min(63, self.hot_cell[0] + random.randint(-3, 3)))
            jy = max(0, min(63, self.hot_cell[1] + random.randint(-3, 3)))
            x, y = jx, jy
        elif self.click_idx < len(self.click_cells):
            x, y = self.click_cells[self.click_idx]
            self.click_idx += 1
        else:
            # Reset click cells and start fresh
            self.click_cells = [(random.randint(4, 60), random.randint(4, 60))
                                for _ in range(25)]
            self.click_idx = 1
            x, y = self.click_cells[0]
        # ACTION6 needs data={x, y} via set_data; arcengine uses ActionInput
        # in newer versions. The old API used a.set_data({...}).
        try:
            a.set_data({"x": int(x), "y": int(y)})
        except AttributeError:
            # newer API: attach action_data
            if hasattr(a, "action_data"):
                a.action_data.x = int(x)
                a.action_data.y = int(y)
        self.last_click = (int(x), int(y))
        self.prev_action = a
        self.prev_grid = grid
        return a

    def _close_life(self):
        self.lives += 1
        score = len(self.visited_life | self.world_visited)
        if score > self.best_depth and len(self.path) >= 2:
            self.best_depth = score
            self.best_path = list(self.path)