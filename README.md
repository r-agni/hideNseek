# Hide and Seek Robot (Unitree G1 + Isaac Lab)

## Goal
Train two Unitree G1 agents in a 3D indoor environment:
- `hider`: gets a short hiding phase, then survives without being seen
- `seeker`: searches and detects the hider

The project focuses on simulation-first multi-agent RL with room-scale scenes (Replica) and supports optional LLM-based action policies for experimentation.

## Environment
- Simulator stack: Isaac Sim + Isaac Lab
- Robot asset: Unitree G1 (`G129_CFG_WITH_DEX3_BASE_FIX`)
- Scene: Replica `apartment_0` converted to USD
- Sensors: egocentric cameras attached to each robot
- Runtime target: native Ubuntu or native Windows with working NVIDIA Vulkan/CUDA

## Game / Training Policy
Each step uses a 4D action:
- seeker: `[forward, turn]`
- hider: `[forward, turn]`

Game phases:
1. `INIT`
2. `HIDING` (seeker frozen, hider moves)
3. `SEEKING` (both can move)
4. `DONE`

Current policy options:
- random velocity baselines
- optional OpenAI-compatible LLM velocity policy (`NEBIUS_API_KEY`)

Self-play loop is implemented in:
- `hide_and_seek/training/self_play.py`
- `scripts/train_self_play.py`

## Setup
```bash
conda env create -f environment.yml
conda activate hide-and-seek
pip install -e .
```

If you use Unitree configs from `~/unitree_sim_isaaclab`, set:
```bash
export PROJECT_ROOT=~/unitree_sim_isaaclab
```

## API Key
Local `.env` is used for secrets:
- `NEBIUS_API_KEY=...`

Load before running LLM policy commands:
```bash
set -a
source .env
set +a
```

## Data Pipeline (Replica)
Raw scene location:
- `data/scenes/replica_raw/apartment_0`

Convert to USD:
```bash
conda run -n hide-and-seek python scripts/convert_replica.py \
  --scene-dir data/scenes/replica_raw/apartment_0 \
  --output-dir data/scenes/replica_usd/apartment_0
```

Expected output:
- `data/scenes/replica_usd/apartment_0/mesh.usd`

## Run Commands
Regression tests:
```bash
TMPDIR=/tmp conda run -n hide-and-seek python -m pytest \
  tests/test_game_logic.py tests/test_replica_scene.py -v
```

Environment smoke test:
```bash
conda run -n hide-and-seek python scripts/test_env.py --headless
```

Self-play rollout:
```bash
conda run -n hide-and-seek python scripts/train_self_play.py --headless --steps 2000
```

Enable LLM policy for one role:
```bash
conda run -n hide-and-seek python scripts/train_self_play.py --headless --hider-llm
```

## Important Config Knobs
- Scene override:
  - `HNS_REPLICA_SCENE_USD=/abs/path/to/mesh.usd`
- Default scene path is set in:
  - `hide_and_seek/env/hide_and_seek_env_cfg.py`
- Camera prim paths are configured in:
  - `HideAndSeekSceneCfg.seeker_camera`
  - `HideAndSeekSceneCfg.hider_camera`

## Repo Layout
- `hide_and_seek/env/` - env config, actions, observations, rewards, terminations
- `hide_and_seek/game/` - phase manager and visibility logic
- `hide_and_seek/utils/` - scene conversion and utilities
- `hide_and_seek/training/` - self-play and policy adapters
- `scripts/` - conversion, env tests, rollout entrypoints
- `tests/` - unit/regression tests

## Notes
- `.env` is gitignored.
- For Isaac Sim startup issues in WSL (Vulkan/CUDA enumeration), use native GPU environments.
