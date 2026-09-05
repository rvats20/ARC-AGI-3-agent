"""One-off: play ls20 with a scripted sequence and dump frames to see what
the agent actually observes (frame format, player movement)."""
import sys, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "ARC-AGI-3-Agents"))

from arcengine import GameAction, GameState
import importlib.util
spec = importlib.util.spec_from_file_location("ua", ROOT / "agent" / "my_agent.py")
ua = importlib.util.module_from_spec(spec); spec.loader.exec_module(ua)

# reuse play_local's harness pieces minimally: use its main funcs
sys.argv = ["play_local.py", "--game", "ls20", "--max-steps", "50"]
import scripts.play_local as pl

# Need to patch the MyAgent that play_local actually uses (it does its own import)
frames_dump = []
orig_load = pl.load_my_agent_class
def patched_load():
    MyAgentCls = orig_load()
    orig_choose = MyAgentCls.choose_action
    def patched_choose(self, frames, latest):
        if len(frames_dump) < 40:
            g = ua._grid(latest.frame) if latest.frame else None
            # Use diff-based centroid tracking like the agent does
            pos = None
            if self.prev_grid is not None and g is not None:
                d = ua.diff_cells(self.prev_grid, g)
                if d:
                    pos = ua.centroid(d)
            frames_dump.append({
                "state": str(latest.state), "levels": latest.levels_completed,
                "pos": list(pos) if pos else None,
                "shape": [len(g), len(g[0])] if g else None,
                "uniq": sorted({v for row in g for v in row})[:12] if g else None,
            })
        return orig_choose(self, frames, latest)
    MyAgentCls.choose_action = patched_choose
    return MyAgentCls
pl.load_my_agent_class = patched_load

try:
    pl.main()
except SystemExit:
    pass
print(json.dumps(frames_dump[:30], indent=0))
