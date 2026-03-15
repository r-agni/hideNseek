"""Test dual-agent environment functionality.

Run with:
    isaaclab -p -m pytest tests/test_dual_agent.py -v
"""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture(scope="module")
def env():
    """Create the hide-and-seek environment (module-scoped)."""
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    app = launcher.app

    from hide_and_seek.env.hide_and_seek_env import HideAndSeekEnv
    from hide_and_seek.env.hide_and_seek_env_cfg import HideAndSeekEnvCfg

    cfg = HideAndSeekEnvCfg()
    cfg.scene.num_envs = 2
    environment = HideAndSeekEnv(cfg)

    yield environment

    environment.close()
    app.close()


def test_reset_returns_observations(env):
    """Reset should return valid observation dict."""
    obs, info = env.reset()
    assert obs is not None
    assert isinstance(obs, dict)


def test_agents_at_different_positions(env):
    """Seeker and hider should spawn at different positions."""
    import torch

    env.reset()
    seeker_pos = env.scene["seeker"].data.root_pos_w
    hider_pos = env.scene["hider"].data.root_pos_w
    dist = torch.norm(seeker_pos - hider_pos, dim=-1)
    assert (dist > 0).all(), "Agents should be at different positions"


def test_step_returns_correct_shapes(env):
    """Step should return properly shaped tensors."""
    import torch

    env.reset()
    action = torch.zeros(env.num_envs, 4, device=env.device)
    obs, reward, terminated, truncated, info = env.step(action)

    assert terminated.shape == (env.num_envs,)
    assert truncated.shape == (env.num_envs,)


def test_phase_manager_initialized(env):
    """Phase manager should be initialized after reset."""
    env.reset()
    assert env.phase_manager is not None
    assert env.phase_manager.num_envs == env.num_envs


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
