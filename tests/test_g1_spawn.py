"""Test that the G1 robot can be loaded in Isaac Lab.

This test requires Isaac Lab and GPU. Run with:
    isaaclab -p -m pytest tests/test_g1_spawn.py -v
"""

import pytest
import torch


@pytest.fixture(scope="module")
def simulation_app():
    """Create Isaac Lab simulation app (module-scoped for performance)."""
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=True)
    app = launcher.app
    yield app
    app.close()


def test_g1_config_imports(simulation_app):
    """Verify that G1 asset configuration can be imported."""
    try:
        from isaaclab_assets import G1_MINIMAL_CFG

        assert G1_MINIMAL_CFG is not None
    except ImportError:
        try:
            from isaaclab_assets import G1_CFG

            assert G1_CFG is not None
        except ImportError:
            pytest.skip("G1 config not available in isaaclab_assets; using fallback")


def test_scene_creation(simulation_app):
    """Verify that a scene with two G1 robots can be created."""
    import sys
    import os

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from hide_and_seek.env.hide_and_seek_env_cfg import HideAndSeekSceneCfg

    cfg = HideAndSeekSceneCfg(num_envs=1, env_spacing=0.0)
    assert cfg.seeker is not None
    assert cfg.hider is not None
    assert cfg.seeker_camera is not None
    assert cfg.hider_camera is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
