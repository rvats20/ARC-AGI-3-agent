"""Splice the current `agent/my_agent.py` into `notebooks/submission.ipynb`.

The notebook follows the exact pattern used by Kaggle's official sample
("ARC3 Sample Submission - Stochastic Goose"):

  Cell 1: install the `arc-agi` wheel from the offline competition dataset.
  Cell 2: write `my_agent.py` to /kaggle/working/ — its body is THIS file.
  Cell 3: if running inside the Kaggle competition rerun, wait for the
          gateway sidecar, copy the framework into /kaggle/working/, register
          MyAgent, and run `python main.py --agent myagent`.
  Cell 4: otherwise (during commit / save-and-run-all), write a dummy
          submission.parquet so Kaggle accepts the commit.

You don't normally need to call this directly — `make submit` runs it for you.
"""
from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent

# ─────────────────────────────────────────────────────────────────────────────
# CHANGE THIS ONE LINE TO PICK YOUR KAGGLE ACCELERATOR
# Options:
#   "cpu"      — no GPU. Good for the random starter or any non-ML agent.
#   "t4"       — Nvidia T4 ×2 (default; matches Kaggle's sample submission).
#   "p100"     — Nvidia P100 (single big-memory GPU).
#   "rtx6000"  — Nvidia RTX 6000 (g4-standard-48). ARC-AGI-3 exclusive,
#                burns GPU quota faster — use only when you're confident.
# ─────────────────────────────────────────────────────────────────────────────
ACCELERATOR = "rtx6000"

# Internal mapping; don't edit unless Kaggle adds new options.
_ACCELERATORS = {
    "cpu":     {"name": "none",            "gpu": False},
    "t4":      {"name": "nvidiaTeslaT4",   "gpu": True},
    "p100":    {"name": "nvidiaTeslaP100", "gpu": True},
    "rtx6000": {"name": "nvidiaRtx6000",   "gpu": True},
}

ROOT = Path(__file__).resolve().parents[1]
AGENT_SRC = ROOT / "agent" / "my_agent.py"
NOTEBOOK_PATH = ROOT / "notebooks" / "submission.ipynb"
METADATA_PATH = ROOT / "notebooks" / "kernel-metadata.json"


def code_cell(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {"trusted": True},
        "outputs": [],
        "execution_count": None,
        "source": source,
    }


def markdown_cell(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def build() -> dict:
    if not AGENT_SRC.exists():
        raise SystemExit(f"Could not find {AGENT_SRC}")
    agent_body = AGENT_SRC.read_text()

    install_cell = code_cell(
        "!pip install --no-index --find-links \\\n"
        "    /kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels \\\n"
        "    arc-agi python-dotenv"
    )

    # We write the agent to /tmp/ (not /kaggle/working/) so it does NOT appear
    # as a notebook output. Otherwise the "Submit to Competition" UI would
    # offer it as a candidate submission file alongside submission.parquet,
    # and an unlucky default selection rejects the submission.
    write_agent_cell = code_cell(
        "%%writefile /tmp/my_agent.py\n" + agent_body
    )

    run_cell_source = dedent(
        """\
        import os

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            # Wait for the gateway sidecar to be ready.
            !curl --fail --retry 999 --retry-all-errors --retry-delay 5 \\
                  --retry-max-time 600 http://gateway:8001/api/games

            # Copy the framework into a writable location.
            !cp -r /kaggle/input/competitions/arc-prize-2026-arc-agi-3/ARC-AGI-3-Agents \\
                   /kaggle/working/ARC-AGI-3-Agents

            # Drop our agents in as framework templates.
            !cp /tmp/my_agent.py \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/my_agent.py
            !cp /tmp/qwen_agent.py \\
                /kaggle/working/ARC-AGI-3-Agents/agents/templates/qwen_agent.py

            # Register agents in the framework's agent registry. We rewrite
            # __init__.py because the upstream version eagerly imports
            # templates with deps we don't ship (langgraph, smolagents, etc.).
            # QwenAgent subclasses the heuristic MyAgent and falls back to it
            # when no vLLM endpoint is reachable.
            with open('/kaggle/working/ARC-AGI-3-Agents/agents/__init__.py', 'w') as f:
                f.write(\"\"\"from typing import Type
        from dotenv import load_dotenv
        from .agent import Agent, Playback
        from .swarm import Swarm
        from .templates.random_agent import Random
        from .templates.my_agent import MyAgent
        from .templates.qwen_agent import QwenAgent

        load_dotenv()

        AVAILABLE_AGENTS: dict[str, Type[Agent]] = {
            'random': Random,
            'myagent': QwenAgent,
        }
        \"\"\")

            # Point the framework at the gateway sidecar.
            with open('/kaggle/working/ARC-AGI-3-Agents/.env', 'w') as f:
                f.write(\"\"\"SCHEME=http
        HOST=gateway
        PORT=8001
        ARC_API_KEY=test-key-123
        ARC_BASE_URL=http://gateway:8001/
        OPERATION_MODE=online
        ENVIRONMENTS_DIR=
        RECORDINGS_DIR=/kaggle/working/server_recording
        \"\"\")

            # Run it. The gateway records every action and emits submission.parquet.
            # QWEN_AGENT=1 enables the VLM agent; it self-falls-back to the
            # heuristic explorer if the vLLM endpoint is not serving.
            !cd /kaggle/working/ARC-AGI-3-Agents && \\
                MPLBACKEND=agg \\
                QWEN_AGENT=1 \\
                QWEN_BASE_URL=${QWEN_BASE_URL:-http://localhost:8000/v1} \\
                QWEN_MODEL=${QWEN_MODEL:-Qwen/Qwen3-VL} \\
                python main.py --agent myagent
        """
    )
    run_cell = code_cell(run_cell_source)

    # Launch a local vLLM OpenAI-compatible server from the model dataset that
    # is mounted as a Kaggle input. The competition rerun has no internet, so
    # the model weights + vLLM wheels must come from datasets. We auto-detect
    # them by scanning /kaggle/input/* (works regardless of the exact slug you
    # attached): the model dir holds config.json + *.safetensors; the wheelhouse
    # holds a wheels/ folder. If nothing is mounted, QwenAgent self-falls back
    # to the heuristic explorer.
    serve_cell_source = dedent(
        """\
        import os, subprocess, glob, time

        def _find_model_dir():
            for d in glob.glob('/kaggle/input/*'):
                if os.path.isfile(os.path.join(d, 'config.json')) and \\
                   (glob.glob(os.path.join(d, '*.safetensors')) or
                    glob.glob(os.path.join(d, 'layers-*.safetensors'))):
                    return d
            return None

        def _find_wheelhouse():
            for d in glob.glob('/kaggle/input/*'):
                if os.path.isdir(os.path.join(d, 'wheels')):
                    return d
            return None

        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            model_dir = _find_model_dir()
            wh = _find_wheelhouse()
            if not model_dir:
                print('No model dataset mounted under /kaggle/input -> '
                      'QwenAgent will use the heuristic fallback.')
            else:
                print('Model dir:', model_dir, '| wheelhouse:', wh)
                if wh:
                    !pip install --no-index --find-links {wh}/wheels vllm 2>&1 | tail -3
                # vLLM registers the model under the --model path; export it so
                # the agent (run in the next cell, same kernel) requests the
                # exact same model id.
                os.environ['QWEN_MODEL'] = model_dir
                os.environ['QWEN_BASE_URL'] = 'http://localhost:8000/v1'
                serve_proc = subprocess.Popen([
                    'python', '-m', 'vllm.entrypoints.openai.api_server',
                    '--model', model_dir,
                    '--tensor-parallel-size', os.getenv('VLLM_TP', '1'),
                    '--max-model-len', os.getenv('VLLM_MAX_LEN', '8192'),
                    '--dtype', 'auto', '--port', '8000',
                    '--enable-auto-tool-choice',
                ], env={**os.environ, 'VLLM_MODEL_DIR': model_dir})
                import urllib.request
                up = False
                for _ in range(180):
                    try:
                        urllib.request.urlopen('http://localhost:8000/v1/models', timeout=2)
                        up = True
                        break
                    except Exception:
                        time.sleep(5)
                print('vLLM server up:', up if up else
                      'did NOT come up in time; agent falls back to heuristic')
                if not up:
                    serve_proc.terminate()
        """
    )
    serve_cell = code_cell(serve_cell_source)

    dummy_submission_cell = code_cell(
        dedent(
            """\
            import os
            if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
                # Save-and-run-all (commit) mode: emit a dummy submission so the
                # commit succeeds. The real submission.parquet is produced by the
                # gateway during competition rerun.
                import pandas as pd
                submission = pd.DataFrame(
                    data=[['1_0', '1', True, 1]],
                    columns=['row_id', 'game_id', 'end_of_game', 'score'])
                submission.to_parquet('/kaggle/working/submission.parquet', index=False)
                submission.head()
            """
        )
    )

    if ACCELERATOR not in _ACCELERATORS:
        raise SystemExit(
            f"Unknown ACCELERATOR={ACCELERATOR!r}. Pick one of: "
            f"{sorted(_ACCELERATORS)}"
        )
    accel = _ACCELERATORS[ACCELERATOR]

    notebook = {
        "metadata": {
            "kernelspec": {
                "language": "python",
                "display_name": "Python 3",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "mimetype": "text/x-python",
                "file_extension": ".py",
                "pygments_lexer": "ipython3",
            },
            "kaggle": {
                "accelerator": accel["name"],
                "isInternetEnabled": False,
                "isGpuEnabled": accel["gpu"],
                "language": "python",
                "sourceType": "notebook",
            },
        },
        "nbformat_minor": 4,
        "nbformat": 4,
        "cells": [
            markdown_cell(
                "# ARC Prize 2026 — ARC-AGI-3 Submission\n\n"
                "Built from `agent/my_agent.py` via `scripts/build_notebook.py`. "
                "Do not edit cells directly — edit the source file and re-run "
                "`make submit`."
            ),
            install_cell,
            write_agent_cell,
            run_cell,
            serve_cell,
            dummy_submission_cell,
        ],
    }
    return notebook


def main() -> None:
    NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
    NOTEBOOK_PATH.write_text(json.dumps(build(), indent=1))
    print(f"[build_notebook] Wrote {NOTEBOOK_PATH.relative_to(ROOT)}  "
          f"(accelerator: {ACCELERATOR})")

    # Keep notebooks/kernel-metadata.json in sync so the user never has to
    # edit it just to flip CPU ↔ GPU.
    if METADATA_PATH.exists():
        meta = json.loads(METADATA_PATH.read_text())
        wanted = _ACCELERATORS[ACCELERATOR]["gpu"]
        if meta.get("enable_gpu") != wanted:
            meta["enable_gpu"] = wanted
            METADATA_PATH.write_text(json.dumps(meta, indent=2) + "\n")
            print(f"[build_notebook] Synced enable_gpu={wanted} in "
                  f"{METADATA_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
