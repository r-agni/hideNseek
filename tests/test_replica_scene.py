"""Test Replica scene conversion utilities.

These tests don't require Isaac Lab — they test the PLY→OBJ pipeline.
"""

import os
import tempfile

import numpy as np
import pytest
import trimesh

from hide_and_seek.utils.scene_converter import convert_ply_to_obj


def test_convert_ply_to_obj():
    """Test PLY to OBJ conversion with a synthetic mesh."""
    # Create a simple test mesh
    vertices = np.array(
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
    )
    faces = np.array([[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]])
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    with tempfile.TemporaryDirectory() as tmpdir:
        ply_path = os.path.join(tmpdir, "test.ply")
        obj_path = os.path.join(tmpdir, "output", "test.obj")

        # Save as PLY
        mesh.export(ply_path)

        # Convert
        result = convert_ply_to_obj(ply_path, obj_path)

        assert os.path.exists(result)
        # Load back and verify
        loaded = trimesh.load(result)
        assert len(loaded.faces) == 4
        assert len(loaded.vertices) == 4


def test_convert_missing_file():
    """Should raise FileNotFoundError for missing PLY."""
    with pytest.raises(Exception):
        convert_ply_to_obj("/nonexistent/mesh.ply", "/tmp/out.obj")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
