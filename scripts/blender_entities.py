"""Blender-only Object Index export; source .blend files are never saved.

Cycles Object Index is a hard first-surface data pass, not an antialiased
Cryptomatte. Alpha threshold 0.5 skips sufficiently transparent alpha surfaces;
transmission glass with opaque alpha can still own the visible object label.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile
import time

import bpy
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"src"))
sys.path.insert(0, str(ROOT/"scripts"))
from blender_export import camera_data, inspect_scene, set_frame
from scene_reconstruction.blender_entities import (
    decode_index_pass, validate_entities, validate_selection, visibility_statistics,
)
from scene_reconstruction.blender_export import pixel_resize_affine
from scene_reconstruction.common import sha256, write_json

GEOMETRY_TYPES = {"MESH", "CURVE", "SURFACE", "FONT", "META", "VOLUME", "POINTCLOUD"}


def set_device(scene, device):
    scene.render.engine = "CYCLES"
    if device == "CPU":
        scene.cycles.device = "CPU"
        return ["CPU"]
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = device
    prefs.get_devices()
    selected = []
    for entry in prefs.devices:
        entry.use = entry.type == device
        if entry.use:
            selected.append(entry.name)
    if not selected:
        raise RuntimeError(f"No Cycles {device} devices are available")
    scene.cycles.device = "GPU"
    return selected


def setup_render(scene, directory, samples, device, alpha_threshold=.5):
    devices = set_device(scene, device)
    scene.cycles.samples = samples
    scene.cycles.use_adaptive_sampling = False
    scene.cycles.use_denoising = False
    scene.cycles.seed = 0
    scene.cycles.use_animated_seed = False
    scene.render.use_persistent_data = True
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
    layer.use_pass_object_index = True
    layer.pass_alpha_threshold = alpha_threshold
    scene.use_nodes = True
    scene.render.use_compositing = True
    nodes, links = scene.node_tree.nodes, scene.node_tree.links
    nodes.clear()
    render = nodes.new("CompositorNodeRLayers")
    render.layer = layer.name
    writer = nodes.new("CompositorNodeOutputFile")
    writer.base_path = str(directory)
    writer.file_slots[0].path = "entity_"
    writer.format.file_format = "OPEN_EXR"
    writer.format.color_depth = "32"
    writer.format.color_mode = "RGB"
    writer.format.exr_codec = "ZIP"
    links.new(render.outputs["IndexOB"], writer.inputs[0])
    composite = nodes.new("CompositorNodeComposite")
    links.new(render.outputs["Image"], composite.inputs["Image"])
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    return devices


def read_labels(directory, width, height, allowed_ids):
    files = list(Path(directory).glob("entity_*.exr"))
    if len(files) != 1:
        raise RuntimeError(f"Expected exactly one Object Index EXR, got {len(files)}")
    image = bpy.data.images.load(str(files[0]), check_existing=False)
    try:
        image.colorspace_settings.name = "Non-Color"
        values = np.empty(len(image.pixels), dtype=np.float32)
        image.pixels.foreach_get(values)
        if tuple(image.size) != (width, height) or values.size != width*height*4:
            raise RuntimeError("Object Index pass pixel grid differs from requested render")
        return decode_index_pass(np.flipud(values.reshape(height, width, 4))[..., 0], allowed_ids)
    finally:
        bpy.data.images.remove(image)
        files[0].unlink()


def assign_entities(scene, entities):
    assigned = validate_entities(entities)
    excluded = set(entities.get("excluded_object_names", []))
    missing = set(assigned) - set(scene.objects.keys())
    if missing:
        raise ValueError(f"Entity objects absent from scene: {sorted(missing)}")
    unassigned = []
    for obj in scene.objects:
        obj.pass_index = assigned.get(obj.name, 0)
        if obj.type in GEOMETRY_TYPES and not obj.hide_render and obj.name not in assigned and obj.name not in excluded:
            unassigned.append(obj.name)
    if unassigned:
        raise ValueError(f"Render-enabled geometry missing entity assignments: {sorted(unassigned)}")
    # Instancers cannot silently inherit a zero ID from geometry outside scene.objects.
    for instance in bpy.context.evaluated_depsgraph_get().object_instances:
        obj = instance.object
        if obj.type in GEOMETRY_TYPES and not obj.hide_render and instance.show_self:
            name = obj.original.name
            if name not in assigned and name not in excluded:
                raise ValueError(f"Evaluated geometry instance lacks an entity assignment: {name}")
    return sorted({entry["id"] for entry in entities["entities"]})


def export(config):
    output = Path(config["output"])
    for key, path_key in [("manifest_sha256", "manifest"), ("selection_sha256", "selection"),
                          ("entities_sha256", "entities"), ("source_blend_sha256", "blend")]:
        if sha256(Path(config[path_key])) != config[key]:
            raise ValueError(f"Input changed before Blender started: {path_key}")
    manifest = json.loads(Path(config["manifest"]).read_text())
    selection = json.loads(Path(config["selection"]).read_text())
    entities = json.loads(Path(config["entities"]).read_text())
    validate_entities(entities, config["source_blend_sha256"])
    frames = validate_selection(selection, manifest, config["manifest_sha256"])
    frames = [frame for frame in frames if frame["index"] in config["render_indices"]]
    bpy.ops.wm.open_mainfile(filepath=config["blend"], load_ui=False)
    scene = bpy.context.scene
    inspection = inspect_scene(scene)
    if inspection["linked_libraries"]:
        raise ValueError("Entity rendering requires a self-contained Blender scene")
    if scene.camera is None:
        raise ValueError("Scene has no active camera")
    allowed = assign_entities(scene, entities)
    source_w, source_h = int(manifest["width"]), int(manifest["height"])
    width, height = max(1, round(source_w*config["resolution_scale"])), max(1, round(source_h*config["resolution_scale"]))
    if abs(source_w/source_h-scene.render.resolution_x/scene.render.resolution_y) > 1e-6:
        raise ValueError("Scene aspect ratio differs from canonical source framing")
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = width, height, 100
    old_aspect_x, old_aspect_y = inspection["pixel_aspect"]
    aspect = (source_w/source_h)/(width/height)*old_aspect_x/old_aspect_y
    scene.render.pixel_aspect_x, scene.render.pixel_aspect_y = max(1., aspect), max(1., 1./aspect)
    pixel_affine = pixel_resize_affine((source_w, source_h), (width, height))
    fps, start = scene.render.fps/scene.render.fps_base, scene.frame_start
    (output/"frames").mkdir(exist_ok=True)
    if config["rgb"]:
        (output/"rgb").mkdir(exist_ok=True)
    index = {"schema_version": 1, "stage": "blender_entities", "status": "running",
             "source_manifest_sha256": config["manifest_sha256"], "manifest_sha256": config["manifest_sha256"],
             "selection_sha256": config["selection_sha256"], "entities_sha256": config["entities_sha256"],
             "source_blend_sha256": config["source_blend_sha256"], "source_blend": config["blend"],
             "width": width, "height": height, "source_width": source_w, "source_height": source_h,
             "source_to_export_pixel_affine": pixel_affine.tolist(),
             "frame_count": len(frames), "source_frame_count": len(manifest["frames"]),
             "entity_count": len(entities["entities"]), "entities": entities["entities"],
             "excluded_object_names": entities.get("excluded_object_names", []),
             "label_semantics": "uint16 visible-surface entity IDs; 0 is background or explicitly excluded geometry; top-left pixel origin",
             "camera_convention": "OpenCV camera basis; camera_to_world maps camera coordinates into unchanged Blender world",
             "inspection": inspection, "blender_version": bpy.app.version_string,
             "timeline": {"formula": "scene.frame_start + source_time_seconds * scene_fps", "frame_start": start,
                          "scene_fps": fps, "outside_scene_frame_range": []},
             "rendering": {"engine": "CYCLES", "device": config["render_device"], "samples": config["samples"],
                           "pass": "Object Index / IndexOB", "anti_aliasing": "ID pass is not antialiased; hard integer labels",
                           "pass_alpha_threshold": .5, "motion_blur": False, "depth_of_field": False,
                           "adaptive_sampling": False, "denoising": False,
                           "transparency": "Alpha below 0.5 is skipped by the data pass; transmission with opaque alpha may retain the front object's ID. No fractional or overlapping labels.",
                           "compositing": "Direct authored-material RGB plus IndexOB; original compositor replaced in process only",
                           "documentation": "https://docs.blender.org/manual/en/4.4/render/layers/passes.html"},
             "frames": []}
    write_json(output/"index.json", index)
    try:
        with tempfile.TemporaryDirectory(prefix=".entity-pass-", dir=output) as temporary:
            index["rendering"]["devices"] = setup_render(scene, temporary, config["samples"], config["render_device"])
            for position, frame in enumerate(frames):
                tick = time.monotonic()
                blender_frame = start+frame["time_seconds"]*fps
                set_frame(scene, blender_frame)
                # Assignment may be animated in source files; override IDs at every timestamp.
                assign_entities(scene, entities)
                if blender_frame < scene.frame_start or blender_frame > scene.frame_end:
                    index["timeline"]["outside_scene_frame_range"].append(frame["index"])
                if scene.camera is None:
                    raise ValueError("No active camera at selected timestamp")
                intrinsic, matrix, _ = camera_data(scene, scene.camera, width, height)
                rgb_file = f"rgb/{frame['index']:06d}.png"
                if config["rgb"]:
                    scene.render.filepath = str(output/rgb_file)
                print(f"Entity frame {position+1}/{len(frames)}: source {frame['index']} at Blender {blender_frame:.6f}", flush=True)
                bpy.ops.render.render(write_still=config["rgb"])
                labels = read_labels(temporary, width, height, allowed)
                label_file = f"frames/{frame['index']:06d}.npz"
                np.savez_compressed(output/label_file, entity_ids=labels, intrinsics=intrinsic,
                                    camera_to_world=matrix, world_to_camera=np.linalg.inv(matrix))
                row = {"index": frame["index"], "time_seconds": frame["time_seconds"], "pts_seconds": frame.get("pts_seconds"),
                       "blender_frame": blender_frame, "camera_name": scene.camera.name, "entity_file": label_file,
                       "entities": visibility_statistics(labels), "render_seconds": time.monotonic()-tick}
                if config["rgb"]:
                    row["rgb_file"] = rgb_file
                index["frames"].append(row)
                write_json(output/"index.json", index)
        if sha256(Path(config["blend"])) != config["source_blend_sha256"]:
            raise RuntimeError("Original Blender scene changed during rendering")
        index["status"] = "complete"
        write_json(output/"index.json", index)
    except Exception as error:
        index.update(status="failed", error=str(error))
        write_json(output/"index.json", index)
        raise


def validate_pass(output, device="CPU"):
    """Render an independent occlusion, grouping, orientation and alpha fixture."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage = 96, 72, 100
    bpy.ops.object.camera_add(location=(0, 0, 0))
    scene.camera = bpy.context.object
    scene.camera.data.lens = 40
    scene.camera.data.sensor_width = 36
    def plane(name, identifier, location, scale, transparent=False):
        bpy.ops.mesh.primitive_plane_add(size=2, location=location)
        obj = bpy.context.object; obj.name = name; obj.pass_index = identifier
        obj.scale = (scale, scale, scale)
        if transparent:
            material = bpy.data.materials.new(name+" transparent"); material.use_nodes = True
            nodes=material.node_tree.nodes; nodes.clear()
            shader=nodes.new("ShaderNodeBsdfTransparent"); out=nodes.new("ShaderNodeOutputMaterial")
            material.node_tree.links.new(shader.outputs[0], out.inputs[0]); obj.data.materials.append(material)
        return obj
    plane("back", 9, (0, 0, -6), 20)
    plane("group_left", 7, (-.7, .5, -3), .2)
    plane("group_right", 7, (.7, .5, -3), .2)
    plane("front", 1, (-.7, -.5, -3), .2)
    plane("transparent", 11, (.7, -.5, -3), .2, True)
    k, _, _ = camera_data(scene, scene.camera, 96, 72)
    def pixel(x,y,z):
        h=k@np.array([x,-y,-z]); return int(round(h[1]/h[2])),int(round(h[0]/h[2]))
    checks=[]
    with tempfile.TemporaryDirectory(prefix="entity-validation-") as temporary:
        for threshold in [.5, 0.]:
            setup_render(scene, temporary, 1, device, alpha_threshold=threshold)
            bpy.ops.render.render()
            labels=read_labels(temporary, 96, 72, [1,7,9,11])
            expected=[((-0.7,.5,-3),7),((.7,.5,-3),7),((-.7,-.5,-3),1),((.7,-.5,-3),9 if threshold else 11),((0,0,-6),9)]
            for xyz,identifier in expected:
                if labels[pixel(*xyz)] != identifier:
                    raise RuntimeError(f"Synthetic entity label mismatch at {xyz}, threshold {threshold}")
            checks.append({"alpha_threshold":threshold,"passed":True,"visible_ids":np.unique(labels).tolist(),
                           "checks":["two meshes share one entity", "front object occludes background", "top-left image orientation", "transparent alpha threshold"]})
    result={"status":"complete","blender_version":bpy.app.version_string,"device":device,"checks":checks}
    write_json(Path(output),result)
    print(json.dumps(result),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--validate-pass", type=Path)
    parser.add_argument("--render-device", choices=["CPU", "METAL", "CUDA", "OPTIX"], default="CPU")
    args=parser.parse_args(sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else [])
    if args.validate_pass:
        validate_pass(args.validate_pass, args.render_device)
    elif args.config:
        export(json.loads(args.config.read_text()))
    else:
        parser.error("Supply --config or --validate-pass")
