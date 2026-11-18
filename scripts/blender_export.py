"""Run with Blender's Python; originals are opened but never saved."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import tempfile

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scene_reconstruction.blender_export import (
    blender_frame, intrinsics_from_projection, normalize_depth, pixel_resize_affine, ray_norms,
)
from scene_reconstruction.common import sha256, write_json


def set_frame(scene, value):
    integer = math.floor(value)
    scene.frame_set(integer, subframe=value - integer)


def camera_data(scene, camera, width, height):
    graph = bpy.context.evaluated_depsgraph_get()
    evaluated = camera.evaluated_get(graph)
    if evaluated.data.type != "PERSP":
        raise ValueError("Only perspective cameras are supported for metric depth export")
    projection = np.array(evaluated.calc_matrix_camera(
        graph, x=width, y=height,
        scale_x=scene.render.pixel_aspect_x, scale_y=scene.render.pixel_aspect_y))
    intrinsic = intrinsics_from_projection(projection, width, height)
    world = np.array(evaluated.matrix_world.normalized(), dtype=np.float64)
    rotation = world[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5) or np.linalg.det(rotation) < 0:
        raise ValueError(f"Camera {camera.name} has a sheared/reflected transform")
    opencv_c2w = world @ np.diag([1, -1, -1, 1])
    return intrinsic, opencv_c2w, evaluated.data


def setup_render(scene, directory, samples, depth, rgb):
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples if rgb else 1
    scene.cycles.use_denoising = False
    scene.render.use_motion_blur = False
    scene.render.use_border = False
    scene.render.use_crop_to_border = False
    scene.render.use_multiview = False
    scene.render.use_sequencer = False
    for camera in bpy.data.cameras:
        camera.dof.use_dof = False
    layer = bpy.context.view_layer
    for other in scene.view_layers:
        other.use = other == layer
    layer.use_pass_z = depth
    scene.use_nodes = True
    scene.render.use_compositing = True
    nodes, links = scene.node_tree.nodes, scene.node_tree.links
    nodes.clear()
    render = nodes.new("CompositorNodeRLayers")
    render.layer = layer.name
    composite = nodes.new("CompositorNodeComposite")
    links.new(render.outputs["Image"], composite.inputs["Image"])
    if depth:
        writer = nodes.new("CompositorNodeOutputFile")
        writer.base_path = str(directory)
        writer.file_slots[0].path = "depth_"
        writer.format.file_format = "OPEN_EXR"
        writer.format.color_depth = "32"
        writer.format.color_mode = "RGB"
        writer.format.exr_codec = "ZIP"
        links.new(render.outputs["Depth"], writer.inputs[0])
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    return layer


def read_depth_exr(directory, width, height):
    files = list(Path(directory).glob("depth_*.exr"))
    if len(files) != 1:
        raise RuntimeError(f"Expected one depth EXR; found {len(files)}")
    image = bpy.data.images.load(str(files[0]), check_existing=False)
    try:
        image.colorspace_settings.name = "Non-Color"
        values = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(values)
        if tuple(image.size) != (width, height) or values.size != width * height * 4:
            raise RuntimeError("Rendered depth EXR grid is not the requested image size")
        # Blender image buffers are bottom-up; our source rasters are top-down.
        return np.flipud(values.reshape(height, width, 4))[..., 0].copy()
    finally:
        bpy.data.images.remove(image)
        files[0].unlink()


def validate_depth():
    """Measure the actual renderer pass on frontoparallel and tilted planes."""
    reports = []
    convention = None
    with tempfile.TemporaryDirectory(prefix="blender-depth-calibration-") as temporary:
        for angle in [0.0, 0.3]:
            bpy.ops.wm.read_factory_settings(use_empty=True)
            scene = bpy.context.scene
            scene.render.resolution_x, scene.render.resolution_y = 32, 24
            scene.render.resolution_percentage = 100
            bpy.ops.object.camera_add(location=(0, 0, 0))
            camera = bpy.context.object
            camera.data.lens = 30
            scene.camera = camera
            bpy.ops.mesh.primitive_plane_add(size=200, location=(0, 0, -5))
            bpy.context.object.rotation_euler[1] = angle
            setup_render(scene, temporary, 1, True, False)
            intrinsic, _, _ = camera_data(scene, camera, 32, 24)
            bpy.ops.render.render()
            raw = read_depth_exr(temporary, 32, 24)
            y, x = np.mgrid[:24, :32]
            rays = np.stack([x, y, np.ones_like(x)], axis=-1) @ np.linalg.inv(intrinsic).T
            expected = 5 / (1 - np.tan(angle) * rays[..., 0])
            errors = {"camera_z": float(np.max(np.abs(raw - expected))),
                      "ray_distance": float(np.max(np.abs(raw / np.linalg.norm(rays, axis=-1) - expected)))}
            this_convention = min(errors, key=errors.get)
            if errors[this_convention] > 1e-3 or (convention is not None and convention != this_convention):
                raise RuntimeError(f"Cannot establish Blender depth convention: {errors}")
            convention = this_convention
            reports.append({"plane_tilt_radians": angle, "plane_center_z": 5,
                            "max_absolute_z_error": errors[this_convention], "candidate_errors": errors})
    return {"performed": True, "blender_version": bpy.app.version_string,
            "engine": "CYCLES", "raw_pass_convention": convention,
            "tests": reports, "tolerance_blender_units": 0.001,
            "method": "Actual RenderLayers Depth -> float32 EXR -> flipped pixel buffer compared with analytic plane intersections"}


def inspect_scene(scene):
    cameras = []
    for camera in (obj for obj in scene.objects if obj.type == "CAMERA"):
        action = camera.animation_data.action if camera.animation_data else None
        data_action = camera.data.animation_data.action if camera.data.animation_data else None
        cameras.append({"name": camera.name, "projection": camera.data.type,
                        "sensor_fit": camera.data.sensor_fit,
                        "sensor_width_mm": camera.data.sensor_width, "sensor_height_mm": camera.data.sensor_height,
                        "lens_mm_at_open": camera.data.lens,
                        "pose_action_range": list(action.frame_range) if action else None,
                        "lens_action_range": list(data_action.frame_range) if data_action else None})
    images = [{"name": image.name, "filepath": image.filepath,
               "packed": bool(image.packed_file), "source": image.source}
              for image in bpy.data.images if image.source in {"FILE", "MOVIE", "SEQUENCE"}]
    return {"scene_name": scene.name, "frame_start": scene.frame_start, "frame_end": scene.frame_end,
            "fps": scene.render.fps / scene.render.fps_base,
            "original_resolution": [scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage],
            "pixel_aspect": [scene.render.pixel_aspect_x, scene.render.pixel_aspect_y],
            "original_engine": scene.render.engine,
            "original_compositor_nodes": [node.bl_idname for node in scene.node_tree.nodes] if scene.use_nodes and scene.node_tree else [],
            "unit_system": scene.unit_settings.system, "unit_scale_length": scene.unit_settings.scale_length,
            "object_count": len(scene.objects), "cameras": cameras, "images": images,
            "linked_libraries": [library.filepath for library in bpy.data.libraries]}


def export_mesh(output):
    graph = bpy.context.evaluated_depsgraph_get()
    vertices, triangles, ids, colors, objects = [], [], [], [], []
    offset = 0
    for instance in graph.object_instances:
        obj = instance.object
        if obj.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"} or obj.hide_render:
            continue
        if not instance.show_self:
            continue
        mesh = obj.to_mesh(preserve_all_data_layers=False, depsgraph=graph)
        try:
            if mesh is None or not len(mesh.vertices):
                continue
            mesh.calc_loop_triangles()
            if not len(mesh.loop_triangles):
                continue
            positions = np.empty(len(mesh.vertices) * 3, np.float32)
            mesh.vertices.foreach_get("co", positions)
            positions = positions.reshape(-1, 3)
            matrix = np.array(instance.matrix_world, dtype=np.float64)
            # Avoid BLAS floating-status warnings on some macOS bundled NumPy builds.
            positions = (np.einsum("ij,kj->ki", matrix[:3, :3], positions) + matrix[:3, 3]).astype(np.float32)
            if not np.isfinite(positions).all():
                raise ValueError(f"Nonfinite evaluated vertices in {obj.name}")
            faces = np.empty(len(mesh.loop_triangles) * 3, np.int32)
            mesh.loop_triangles.foreach_get("vertices", faces)
            faces = faces.reshape(-1, 3)
            material_indices = np.empty(len(mesh.loop_triangles), np.int32)
            mesh.loop_triangles.foreach_get("material_index", material_indices)
            palette = []
            for material in mesh.materials:
                rgba = list(material.diffuse_color) if material else [0.7, 0.7, 0.7, 1]
                if material and material.use_nodes:
                    principled = next((n for n in material.node_tree.nodes if n.type == "BSDF_PRINCIPLED"), None)
                    if principled is not None:
                        rgba = list(principled.inputs["Base Color"].default_value)
                palette.append(rgba[:3])
            if not palette:
                palette = [[0.7, 0.7, 0.7]]
            palette = np.asarray(palette, dtype=np.float32)
            face_colors = palette[np.clip(material_indices, 0, len(palette) - 1)]
            object_id = len(objects)
            objects.append({"id": object_id, "name": obj.original.name,
                            "instance": bool(instance.is_instance),
                            "bounds_min": positions.min(axis=0).tolist(), "bounds_max": positions.max(axis=0).tolist(),
                            "vertices": len(positions), "triangles": len(faces)})
            vertices.append(positions); triangles.append(faces + offset)
            ids.append(np.full(len(faces), object_id, np.int32)); colors.append(face_colors)
            offset += len(positions)
        finally:
            obj.to_mesh_clear()
    if not vertices:
        raise ValueError("Scene contains no exportable render-enabled mesh geometry")
    np.savez_compressed(output / "geometry.npz", vertices=np.concatenate(vertices), triangles=np.concatenate(triangles),
                        triangle_object_ids=np.concatenate(ids), face_colors=np.concatenate(colors),
                        object_names=np.asarray([item["name"] for item in objects]))
    write_json(output / "objects.json", {"objects": objects})
    return {"file": "geometry.npz", "objects_file": "objects.json", "objects": len(objects),
            "vertices": offset, "triangles": sum(len(value) for value in triangles),
            "scope": "Single evaluated viewport-depsgraph snapshot; hide_render objects excluded. Render-only modifier levels and later object motion may differ from this snapshot.",
            "colors": "Approximate material base colors; texture appearance is in the rendered RGB, not mesh colors"}


def export(config):
    output = Path(config["output"])
    manifest = json.loads(Path(config["manifest"]).read_text())
    if sha256(Path(config["manifest"])) != config["manifest_sha256"]:
        raise ValueError("Source manifest changed before Blender started")
    validation = validate_depth() if config["depth"] else {"performed": False}
    bpy.ops.wm.open_mainfile(filepath=config["blend"], load_ui=False)
    scene = bpy.context.scene
    inspection = inspect_scene(scene)
    if inspection["linked_libraries"]:
        raise ValueError("Linked libraries need a self-contained baseline before export")
    if config.get("camera"):
        camera = scene.objects.get(config["camera"])
        if camera is None or camera.type != "CAMERA":
            raise ValueError(f"Camera not found: {config['camera']}")
        scene.camera = camera
    if scene.camera is None:
        raise ValueError("Blender scene has no active camera")
    source_w, source_h = int(manifest["width"]), int(manifest["height"])
    width = max(1, round(source_w * config["resolution_scale"]))
    height = max(1, round(source_h * config["resolution_scale"]))
    if abs(source_w / source_h - scene.render.resolution_x / scene.render.resolution_y) > 1e-6:
        raise ValueError("Source video and Blender render aspect ratios differ; resolve camera framing before export")
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = width, height, 100
    # Rounded dimensions must not subtly change the physical camera frustum.
    # Adjust pixel aspect so the exported grid is a true full-frame resize.
    old_aspect_x, old_aspect_y = inspection["pixel_aspect"]
    aspect = (source_w / source_h) / (width / height) * old_aspect_x / old_aspect_y
    scene.render.pixel_aspect_x = max(1.0, aspect)
    scene.render.pixel_aspect_y = max(1.0, 1.0 / aspect)
    transform = pixel_resize_affine((source_w, source_h), (width, height))
    fps = scene.render.fps / scene.render.fps_base
    frame_start = float(scene.frame_start if config["frame_start"] is None else config["frame_start"])
    render_set = set(config["render_indices"])
    index = {
        "schema_version": 1, "status": "running", "stage": "blender_export",
        "manifest_sha256": config["manifest_sha256"], "source_manifest": config["manifest"],
        "source_blend": config["blend"], "source_blend_sha256": config["source_blend_sha256"],
        "width": width, "height": height, "source_width": source_w, "source_height": source_h,
        "canonical_width": source_w, "canonical_height": source_h,
        "source_to_export_pixel_affine": transform.tolist(),
        "depth_units": "blender_units", "depth_convention": "camera_z",
        "unit_scale_meters": float(scene.unit_settings.scale_length) if scene.unit_settings.system in {"METRIC", "IMPERIAL"} else None,
        "unit_scale_basis": "Scene unit settings, not an independently measured physical calibration",
        "camera_convention": "OpenCV x right, y down, z forward; camera_to_world maps camera column vectors to Blender world",
        "blender_version": bpy.app.version_string, "inspection": inspection,
        "timeline": {"formula": "frame_start + source_time_seconds * scene_fps * time_scale",
                     "frame_start": frame_start, "scene_fps": fps, "time_scale": config["time_scale"],
                     "outside_scene_frame_range": []},
        "depth_validation": validation,
        "rendering": {"engine": "CYCLES", "device": "CPU", "samples": config["samples"] if config["rgb"] else 1,
                      "export_pixel_aspect": [scene.render.pixel_aspect_x, scene.render.pixel_aspect_y],
                      "depth_of_field": False, "motion_blur": False, "compositing": "Direct RenderLayers; original compositor replaced in process only",
                      "depth_edges": "Single-sample Z pass, not antialiased; no-hit/invalid depth stored as NaN plus validity mask"},
        "frame_count": len(manifest["frames"]), "frames": [],
        "files": {"cameras": "cameras.npz", "geometry": "geometry.npz", "objects": "objects.json"},
    }
    write_json(output / "index.json", index)
    matrices, intrinsics, canonical_intrinsics, lenses, indices, times = [], [], [], [], [], []
    first_frame = frame_start + manifest["frames"][0]["time_seconds"] * fps * config["time_scale"]
    set_frame(scene, first_frame)
    index["geometry"] = {**export_mesh(output), "frame_index": manifest["frames"][0]["index"], "blender_frame": first_frame}
    try:
        with tempfile.TemporaryDirectory(prefix=".render-", dir=output) as temporary:
            if config["depth"] or config["rgb"]:
                layer = setup_render(scene, temporary, config["samples"], config["depth"], config["rgb"])
                index["rendering"]["alpha_threshold"] = layer.pass_alpha_threshold
            for position, frame in enumerate(manifest["frames"]):
                value = frame_start + frame["time_seconds"] * fps * config["time_scale"]
                set_frame(scene, value)
                camera = scene.objects.get(config["camera"]) if config.get("camera") else scene.camera
                if camera is None:
                    raise ValueError("No active camera at a requested source timestamp")
                scene.camera = camera
                intrinsic, matrix, data = camera_data(scene, camera, width, height)
                matrices.append(matrix); intrinsics.append(intrinsic)
                canonical_intrinsics.append(np.linalg.inv(transform) @ intrinsic)
                lenses.append(float(data.lens)); indices.append(frame["index"]); times.append(frame["time_seconds"])
                row = {"index": frame["index"], "time_seconds": frame["time_seconds"],
                       "pts_seconds": frame.get("pts_seconds"), "camera_array_index": position,
                       "blender_frame": value, "camera_name": camera.name}
                if value < scene.frame_start or value > scene.frame_end:
                    index["timeline"]["outside_scene_frame_range"].append(frame["index"])
                if frame["index"] in render_set and (config["depth"] or config["rgb"]):
                    if config["rgb"]:
                        rgb_path = output / "rgb" / f"{frame['index']:06d}.png"
                        rgb_path.parent.mkdir(exist_ok=True)
                        scene.render.filepath = str(rgb_path)
                    print(f"Rendering source frame {frame['index']} at Blender frame {value:.6f}", flush=True)
                    bpy.ops.render.render(write_still=config["rgb"])
                    if config["rgb"]:
                        row["rgb_file"] = str(rgb_path.relative_to(output))
                    if config["depth"]:
                        raw = read_depth_exr(temporary, width, height)
                        depth, valid, radial = normalize_depth(raw, intrinsic, validation["raw_pass_convention"], float(data.clip_end))
                        path = output / "frames" / f"{frame['index']:06d}.npz"
                        path.parent.mkdir(exist_ok=True)
                        np.savez_compressed(path, depth=depth, valid_depth=valid, ray_distance=radial,
                                            intrinsics=intrinsic, camera_to_world=matrix,
                                            world_to_camera=np.linalg.inv(matrix))
                        row.update({"file": str(path.relative_to(output)), "valid_fraction": float(valid.mean()), "depth_units": "blender_units"})
                index["frames"].append(row)
                if frame["index"] in render_set:
                    write_json(output / "index.json", index)
        matrices = np.asarray(matrices)
        np.savez_compressed(output / "cameras.npz", frame_indices=np.asarray(indices, dtype=np.int64),
                            time_seconds=np.asarray(times), camera_to_world=matrices,
                            camera_to_world_blender_basis=matrices @ np.diag([1, -1, -1, 1]),
                            world_to_camera=np.linalg.inv(matrices), intrinsics=np.asarray(intrinsics),
                            intrinsics_source=np.asarray(canonical_intrinsics), intrinsics_canonical=np.asarray(canonical_intrinsics),
                            lens_mm=np.asarray(lenses))
        if sha256(Path(config["blend"])) != config["source_blend_sha256"]:
            raise RuntimeError("Original .blend changed during export")
        index["status"] = "complete"
        index["depth_frame_count"] = sum("file" in frame for frame in index["frames"])
        index["rgb_frame_count"] = sum("rgb_file" in frame for frame in index["frames"])
        write_json(output / "index.json", index)
    except Exception as error:
        index.update({"status": "failed", "error": str(error)})
        write_json(output / "index.json", index)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--validate-depth-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else [])
    if args.validate_depth_only:
        result = validate_depth()
        if args.output:
            write_json(args.output, result)
        print(json.dumps(result, indent=2))
    elif args.config:
        export(json.loads(args.config.read_text()))
    else:
        parser.error("Supply --config or --validate-depth-only")


if __name__ == "__main__":
    main()
