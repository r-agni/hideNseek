"""Convert Replica Dataset PLY meshes to USD format for Isaac Lab."""

import os
import logging

import trimesh

log = logging.getLogger(__name__)


def convert_ply_to_obj(ply_path: str, obj_path: str) -> str:
    """Convert a PLY mesh file to OBJ format using trimesh.

    Args:
        ply_path: Path to input PLY file.
        obj_path: Path to output OBJ file.

    Returns:
        Path to the output OBJ file.
    """
    log.info(f"Converting PLY → OBJ: {ply_path}")
    mesh = trimesh.load(ply_path, process=False)

    # If the result is a Scene (multiple meshes), concatenate them
    if isinstance(mesh, trimesh.Scene):
        meshes = []
        for geom in mesh.geometry.values():
            if isinstance(geom, trimesh.Trimesh):
                meshes.append(geom)
        if meshes:
            mesh = trimesh.util.concatenate(meshes)
        else:
            raise ValueError(f"No valid meshes found in {ply_path}")

    os.makedirs(os.path.dirname(obj_path), exist_ok=True)
    mesh.export(obj_path, file_type="obj")
    log.info(f"Saved OBJ: {obj_path} ({len(mesh.faces)} faces)")
    return obj_path


def convert_obj_to_usd(obj_path: str, usd_dir: str) -> str:
    """Convert an OBJ mesh to USD using Isaac Lab's MeshConverter.

    Args:
        obj_path: Path to input OBJ file.
        usd_dir: Directory to write the output USD file.

    Returns:
        Path to the output USD file.
    """
    from isaaclab.sim.converters import MeshConverter, MeshConverterCfg

    log.info(f"Converting OBJ → USD: {obj_path}")
    cfg = MeshConverterCfg(
        asset_path=os.path.abspath(obj_path),
        usd_dir=os.path.abspath(usd_dir),
        force_usd_conversion=True,
        make_instanceable=False,
        collision_approximation="convexDecomposition",
    )
    converter = MeshConverter(cfg)
    usd_path = converter.usd_path
    log.info(f"Saved USD: {usd_path}")
    return usd_path


def convert_replica_scene(
    scene_dir: str,
    output_dir: str,
    mesh_filename: str = "mesh.ply",
) -> str:
    """Full pipeline: convert a Replica scene from PLY to USD.

    Args:
        scene_dir: Path to Replica scene directory (e.g., data/scenes/replica_raw/apartment_0).
        output_dir: Path to output directory for USD files.
        mesh_filename: Name of the mesh file within the scene directory.

    Returns:
        Path to the output USD file.
    """
    ply_path = os.path.join(scene_dir, mesh_filename)
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"Replica mesh not found: {ply_path}")

    scene_name = os.path.basename(os.path.normpath(scene_dir))

    # Support both styles:
    # 1) output_dir is a parent directory (e.g., .../replica_usd)
    # 2) output_dir is already scene-specific (e.g., .../replica_usd/apartment_0)
    output_dir_norm = os.path.normpath(output_dir)
    if os.path.basename(output_dir_norm) == scene_name:
        scene_output_dir = output_dir_norm
    else:
        scene_output_dir = os.path.join(output_dir_norm, scene_name)

    obj_path = os.path.join(scene_output_dir, "mesh.obj")

    # Step 1: PLY → OBJ
    convert_ply_to_obj(ply_path, obj_path)

    # Step 2: OBJ → USD
    usd_path = convert_obj_to_usd(obj_path, scene_output_dir)

    return usd_path
