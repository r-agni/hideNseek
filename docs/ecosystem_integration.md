# Ecosystem Integration Notes

This project can evolve along two compatible tracks:

1. Isaac Lab track (current codebase): fast robot control prototyping with Unitree G1 assets.
2. Habitat track (alternate backend): rich indoor scenes + multi-agent tasks in Habitat-Sim/Habitat-Lab.

## Key Findings From External Repos

- `facebookresearch/habitat-sim`
  - Physics-enabled simulator with Replica support and URDF robot support.
  - Good fit for scene-heavy embodied tasks.
- `facebookresearch/habitat-lab`
  - Task/training framework on top of Habitat-Sim.
  - Includes multi-agent and RL training baselines.
- `unitreerobotics/unitree_rl_lab`
  - Isaac Lab native Unitree RL environments (Go2/H1/G1).
  - Strong direct compatibility with this repository's Isaac stack.
- `unitreerobotics/unifolm-world-model-action` and `unitreerobotics/unifolm-vla`
  - Useful for policy augmentation (vision-language-action and world-model-assisted control).
  - Best consumed as policy heads or dataset/model pipelines, not as a replacement simulator.

## Recommended Architecture

- Keep this repo as the Isaac-Lab-first training runtime for seeker-vs-hider.
- Add optional policy providers:
  - classic RL policy (e.g. PPO self-play)
  - optional LLM/VLA policy hook for action proposals from egocentric observations
- Later, add a Habitat adapter environment if we want cross-simulator transfer.

## Implemented In This Repo

- `hide_and_seek/training/self_play.py`
  - self-play runner for two-agent loop
  - random baseline velocity policies
  - optional OpenAI-compatible LLM policy scaffold
- `scripts/train_self_play.py`
  - command-line runner for seeker/hider self-play rollouts
  - can enable LLM policy per role with `--seeker-llm` / `--hider-llm`

## Security Note

- Do not hardcode API keys in source.
- Use environment variables (default: `NEBIUS_API_KEY`) and rotate leaked keys immediately.
