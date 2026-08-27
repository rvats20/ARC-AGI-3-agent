"""Qwen3.8 VLM-in-the-loop agent for ARC-AGI-3.

Strategy (matches the milestone-winning "The Duck" / Gemma-4 harnesses):
render the current frame to a PNG, show it to a locally-served multimodal model
via the OpenAI-compatible vLLM endpoint, and have the model return a structured
JSON action. Internet is DISABLED during the Kaggle competition rerun, so the
model must be baked into a dataset / wheelhouse (e.g. "ARC3 vLLM H100
Wheelhouse V3") and served at http://localhost:8000/v1.

Endpoint + model are configurable:
  QWEN_BASE_URL   (default http://localhost:8000/v1)
  QWEN_MODEL      (default Qwen/Qwen3-VL — set to your served model name)
  QWEN_AGENT=1    enable this agent; if the endpoint is unreachable we fall
                  back to the heuristic explorer so the notebook still runs.

The flow is a SINGLE call (action + x,y in one JSON) to save tokens/time under
the 9h / 110-game budget, rather than the official 3-call ask→find→analyze loop.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import time
from typing import Any, Sequence

import numpy as np
from PIL import Image

from arcengine import FrameData, GameAction, GameState

from .my_agent import MyAgent  # heuristic fallback (also has m0r0 solve)

logger = logging.getLogger()

# 16-color ARC palette (RGBA)
_PALETTE = [
    (0xFF, 0xFF, 0xFF, 0xFF), (0xCC, 0xCC, 0xCC, 0xFF), (0x99, 0x99, 0x99, 0xFF),
    (0x66, 0x66, 0x66, 0xFF), (0x33, 0x33, 0x33, 0xFF), (0x00, 0x00, 0x00, 0xFF),
    (0xE5, 0x3A, 0xA3, 0xFF), (0xFF, 0x7B, 0xCC, 0xFF), (0xF9, 0x3C, 0x31, 0xFF),
    (0x1E, 0x93, 0xFF, 0xFF), (0x88, 0xD8, 0xF1, 0xFF), (0xFF, 0xDC, 0x00, 0xFF),
    (0xFF, 0x85, 0x1B, 0xFF), (0x92, 0x12, 0x31, 0xFF), (0x4F, 0xCC, 0x30, 0xFF),
    (0xA3, 0x56, 0xD6, 0xFF),
]
_SCALE = 2
_TARGET = 64 * _SCALE


def grid_to_image(grid: Sequence[Sequence[int]]) -> Image.Image:
    raw = bytearray()
    for row in grid:
        for idx in row:
            raw.extend(_PALETTE[int(idx)])
    img = Image.frombytes("RGBA", (64, 64), bytes(raw))
    return img.resize((_TARGET, _TARGET), Image.NEAREST)


def image_to_base64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _extract_json(content: str) -> dict:
    fence = re.search(r"```json\s*(\{.*?\})\s*```", content, re.S | re.I)
    if fence:
        return json.loads(fence.group(1))
    fence = re.search(r"```\s*(\{.*?\})\s*```", content, re.S)
    if fence:
        return json.loads(fence.group(1))
    s, e = content.find("{"), content.rfind("}")
    if s == -1 or e <= s:
        raise ValueError("no JSON in model reply")
    return json.loads(content[s:e + 1])


SYSTEM_PROMPT = (
    "You are an abstract-reasoning game agent solving turn-based interactive "
    "environments shown to you as pixel-art PNG frames. Games use a 16-color "
    "palette on a 64x64 grid. Your job: look at the frame, infer the rules and "
    "the current goal, and pick the single best next action to make progress "
    "(collect items, avoid hazards, merge targets, reach the exit). Prefer "
    "movement/action before clicking. Return ONLY JSON."
)

ACTION_INSTRUCT = (
    "Available actions: ACTION1=Move Up, ACTION2=Move Down, ACTION3=Move Left, "
    "ACTION4=Move Right, ACTION5=Perform Action, ACTION6=Click object at (x,y) "
    "in exact 0-63 pixel coords, ACTION7=Undo. Respond with JSON only:\n"
    '{"action": "ACTION1", "x": 0, "y": 0, "reasoning": "..."}\n'
    "Use ACTION6 only when clicking is clearly needed; include x,y (0-63). "
    "Otherwise omit x,y. Look carefully at colors and object positions."
)


class QwenAgent(MyAgent):
    """VLM-in-the-loop agent; falls back to the heuristic explorer on any error."""

    MAX_ACTIONS = 2000

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._client = None
        self._model = os.environ.get("QWEN_MODEL", "Qwen/Qwen3-VL")
        self._base_url = os.environ.get("QWEN_BASE_URL", "http://localhost:8000/v1")
        self._enabled = os.environ.get("QWEN_AGENT", "1") == "1"
        self._history: list[dict] = []
        self._last_grid = None
        # probe endpoint reachability once
        if self._enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(base_url=self._base_url,
                                      api_key=os.environ.get("QWEN_API_KEY", "EMPTY"))
                # cheap health check
                self._client.models.list(timeout=5)
                logger.info(f"[QwenAgent] endpoint OK: {self._base_url} model={self._model}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"[QwenAgent] endpoint unreachable ({e}); "
                               f"falling back to heuristic explorer")
                self._client = None

    @property
    def name(self) -> str:
        return f"QwenAgent.{self._model}.{self.MAX_ACTIONS}"

    def _frame_grid(self, frame) -> list:
        g = frame.tolist() if hasattr(frame, "tolist") else frame
        if isinstance(g, (list, tuple)) and len(g) == 1:
            g = g[0]
        # frame is [1][64][64]; take the 64x64
        if isinstance(g, (list, tuple)) and len(g) and isinstance(g[0], (list, tuple)):
            return [[int(v) for v in row] for row in g]
        return [[0] * 64 for _ in range(64)]

    def choose_action(self, frames, latest_frame) -> GameAction:
        # m0r0 is solved deterministically — keep that (free win).
        if str(getattr(self, "game_id", "")).startswith("m0r0"):
            return super().choose_action(frames, latest_frame)
        # NOT_PLAYED / GAME_OVER -> reset (same as heuristic)
        if latest_frame.state in (GameState.NOT_PLAYED, GameState.GAME_OVER):
            self._close_life()
            self.new_life()
            return GameAction.RESET

        if not self._enabled or self._client is None:
            return super().choose_action(frames, latest_frame)

        try:
            grid = self._frame_grid(latest_frame.frame)
            img_b64 = image_to_base64(grid_to_image(grid))
            user_text = ACTION_INSTRUCT
            if self._history:
                user_text = "Previous reasoning: " + self._history[-1][:400] + "\n\n" + ACTION_INSTRUCT
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                    {"type": "text", "text": user_text},
                ]},
            ]
            resp = self._client.chat.completions.create(
                model=self._model, messages=messages,
                temperature=0.2, max_tokens=256, timeout=60,
            )
            action_json = _extract_json(resp.choices[0].message.content)
            action_name = str(action_json.get("action", "")).upper()
            action = GameAction.from_name(action_name)
            if action.is_complex():
                x = max(0, min(int(action_json.get("x", 0)), 63))
                y = max(0, min(int(action_json.get("y", 0)), 63))
                action.set_data({"x": x, "y": y})
            self._history.append(action_json.get("reasoning", ""))
            return action
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[QwenAgent] call failed ({e}); heuristic fallback")
            return super().choose_action(frames, latest_frame)
