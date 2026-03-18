# hideNseek

Two Unitree G1 humanoids learning hide-and-seek in a real apartment scan, from egocentric vision alone.

A testbed for multi-agent RL where neither agent gets a god's-eye view — the seeker has to actually look, and the hider has to reason about what the seeker can see.

## How it works

- **Asymmetric self-play** across four phases: the hider moves alone during `HIDING` while the seeker is frozen, then both move during `SEEKING`. The asymmetry is what makes the task non-trivial — the hider gets information the seeker never had.
- **Egocentric cameras only.** Each robot sees from its own head. No shared world state, no overhead view, so detection is a perception problem rather than a distance check.
- **Real room geometry** — Replica `apartment_0` converted to USD, so occlusion comes from furniture that actually exists rather than synthetic boxes.
- **Unitree G1 humanoids** (`G129_CFG_WITH_DEX3_BASE_FIX`) in Isaac Sim + Isaac Lab, driven by a compact 4D action space of forward and turn velocities per agent.
- **Pluggable policies** — random velocity baselines, or an OpenAI-compatible LLM velocity policy, so learned and prompted behaviour run through the same interface.
- ~3,500 lines across 37 Python modules.

## Run it

```bash
conda env create -f environment.yml
conda activate hide-and-seek
pip install -e .
python scripts/train_self_play.py
```

Needs native Ubuntu or Windows with working NVIDIA Vulkan/CUDA.

Scene conversion, LLM policy setup and the full training notes: [docs/README-full.md](docs/README-full.md).
