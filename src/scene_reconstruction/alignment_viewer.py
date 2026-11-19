"""Create a dependency-free, offline WebGL viewer for a scene comparison."""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import load_manifest, sha256


def sample_mesh(vertices, triangles, object_ids, max_triangles):
    vertices, triangles = np.asarray(vertices), np.asarray(triangles)
    object_ids = np.asarray(object_ids)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Mesh vertices must be finite Nx3")
    if triangles.ndim != 2 or triangles.shape[1] != 3 or not np.issubdtype(triangles.dtype, np.integer):
        raise ValueError("Mesh triangles must be integer Mx3")
    if triangles.size and (triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError("Mesh triangle index out of bounds")
    if (object_ids.shape != (len(triangles),) or not np.issubdtype(object_ids.dtype, np.integer)
            or np.any(object_ids < 0)):
        raise ValueError("Mesh object IDs must be one nonnegative integer per triangle")
    if max_triangles < 1:
        raise ValueError("max_triangles must be positive")
    indices = np.arange(len(triangles))
    if len(indices) > max_triangles:
        indices = np.unique(np.linspace(0, len(indices) - 1, max_triangles).astype(int))
    selected = triangles[indices]
    used, inverse = np.unique(selected.reshape(-1), return_inverse=True)
    return vertices[used], inverse.reshape(-1, 3), object_ids[indices], indices


def _thumbnail(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((270, 360))
        stream = io.BytesIO()
        image.save(stream, format="JPEG", quality=78)
    return "data:image/jpeg;base64," + base64.b64encode(stream.getvalue()).decode("ascii")


def _cameras(path, intrinsics_key):
    with np.load(path, allow_pickle=False) as data:
        result = {key: data[key] for key in ("frame_indices", "time_seconds", "camera_to_world")}
        if intrinsics_key not in data:
            raise ValueError(f"Viewer needs {intrinsics_key} in {path} to use the correct image grid")
        result["intrinsics"] = data[intrinsics_key]
    ids, times, poses, intrinsics = (result[key] for key in
                                     ("frame_indices", "time_seconds", "camera_to_world", "intrinsics"))
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer) or not len(ids)
            or len(set(ids.tolist())) != len(ids) or np.any(ids < 0)):
        raise ValueError("Viewer camera frame IDs must be unique nonnegative integers")
    if times.shape != ids.shape or poses.shape != (len(ids), 4, 4) or intrinsics.shape != (len(ids), 3, 3):
        raise ValueError("Viewer camera arrays have inconsistent dimensions")
    if not all(np.isfinite(value).all() for value in (times, poses, intrinsics)):
        raise ValueError("Viewer camera arrays contain nonfinite values")
    rotations = poses[:, :3, :3]
    if (not np.allclose(poses[:, 3], [0, 0, 0, 1], atol=1e-5, rtol=0)
            or not np.allclose(rotations.swapaxes(1, 2) @ rotations, np.eye(3), atol=1e-4, rtol=0)
            or not np.allclose(np.linalg.det(rotations), 1, atol=1e-4, rtol=0)):
        raise ValueError("Viewer camera poses must have proper rigid rotations")
    if (np.any(intrinsics[:, 0, 0] <= 0) or np.any(intrinsics[:, 1, 1] <= 0)
            or not np.allclose(intrinsics[:, 2], [0, 0, 1], atol=1e-6, rtol=0)):
        raise ValueError("Viewer camera intrinsics must be pinhole calibration matrices")
    return result


def _validate_camera_records(cameras, index, manifest):
    records = index.get("frames", [])
    by_id = {record["index"]: record for record in records}
    ids = cameras["frame_indices"].tolist()
    if len(by_id) != len(records) or set(by_id) != set(ids):
        raise ValueError("Viewer index frame IDs differ from its camera archive")
    source_records = {record["index"]: record for record in manifest["frames"]}
    for i, (frame_id, time) in enumerate(zip(ids, cameras["time_seconds"])):
        if frame_id not in source_records:
            raise ValueError(f"Viewer camera frame {frame_id} is absent from the source manifest")
        record = by_id[frame_id]
        for expected in (record["time_seconds"], source_records[frame_id]["time_seconds"]):
            if not np.isfinite(expected) or abs(float(expected) - time) > 1e-6:
                raise ValueError(f"Viewer camera timestamp differs from the source manifest/index at frame {frame_id}")
        if "camera_array_index" in record and record["camera_array_index"] != i:
            raise ValueError(f"Viewer camera_array_index is inconsistent at frame {frame_id}")


def pack_cloud(cloud, transform):
    """Keep PLY coordinates losslessly; apply the shared similarity in WebGL.

    Binary arrays avoid millions of Python/JavaScript array objects. Compute
    aligned bounds in bounded chunks rather than retaining a second full cloud.
    """
    from .alignment import transform_points, validate_transform
    transform = validate_transform(transform)
    points = np.asarray(cloud["points"], dtype="<f4")
    colors = np.asarray(cloud["colors"], dtype=np.uint8)
    frame_ids = np.asarray(cloud["frame_indices"], dtype="<i4")
    count = len(points)
    if points.shape != (count, 3) or colors.shape != (count, 3) or frame_ids.shape != (count,):
        raise ValueError("Cloud arrays must have matching point counts")
    if not np.isfinite(points).all():
        raise ValueError("Cloud positions must be finite")
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = transform["scale"] * np.asarray(transform["rotation"])
    matrix[:3, 3] = transform["translation"]
    bounds = None
    if count:
        low, high = np.full(3, np.inf), np.full(3, -np.inf)
        for start in range(0, count, 250_000):
            aligned = transform_points(points[start:start + 250_000], transform)
            if not np.isfinite(aligned).all():
                raise ValueError("Aligned cloud positions must be finite")
            low = np.minimum(low, aligned.min(axis=0))
            high = np.maximum(high, aligned.max(axis=0))
        bounds = [low.tolist(), high.tolist()]
    encode = lambda values: base64.b64encode(np.ascontiguousarray(values).tobytes()).decode("ascii")
    return {"count": count, "encoding": "base64-little-endian",
            "positionsF32": encode(points), "colorsU8": encode(colors),
            "frameIndicesI32": encode(frame_ids),
            "matrix": matrix.T.reshape(-1).tolist(), "bounds": bounds}


def construction_note(manifest_path, blender_index):
    """Find registered construction context by scene hash, not output name."""
    scene_hash = blender_index.get("source_blend_sha256")
    if not scene_hash:
        return ""
    for parent in Path(manifest_path).resolve().parents:
        registry = parent / "blender_scenes" / "baselines.json"
        if registry.is_file():
            for scene in json.loads(registry.read_text()).get("baselines", []):
                if scene.get("sha256") == scene_hash:
                    return scene.get("viewer_description", "")
            break
    return ""


def build_payload(args):
    source_path = Path(args.manifest).resolve()
    manifest = load_manifest(source_path)
    source_hash = sha256(source_path)
    blender = Path(args.blender_export).resolve()
    bi = json.loads((blender / "index.json").read_text())
    if bi.get("schema_version") != 1 or bi.get("status") != "complete" or bi.get("manifest_sha256") != source_hash:
        raise ValueError("Blender export must be complete and match the frame manifest")
    canonical_size = [bi.get("canonical_width", bi.get("source_width")),
                      bi.get("canonical_height", bi.get("source_height"))]
    if canonical_size != [manifest["width"], manifest["height"]]:
        raise ValueError("Blender canonical camera dimensions differ from the source manifest")
    if not all(isinstance(bi.get(key), int) and bi[key] > 0 for key in ("width", "height")):
        raise ValueError("Blender render dimensions must be positive integers")
    bc = _cameras(blender / "cameras.npz", "intrinsics")
    _validate_camera_records(bc, bi, manifest)
    with np.load(blender / "geometry.npz", allow_pickle=False) as mesh:
        total_triangles = len(mesh["triangles"])
        ids = mesh["triangle_object_ids"] if "triangle_object_ids" in mesh else np.zeros(total_triangles, dtype=int)
        vertices, triangles, ids, selected = sample_mesh(mesh["vertices"], mesh["triangles"], ids, args.max_triangles)
        names = mesh["object_names"].tolist() if "object_names" in mesh else ["Scene"]
        if ids.size and ids.max() >= len(names):
            raise ValueError("Mesh object IDs refer to missing object names")
        colors = mesh["face_colors"][selected, :3] if "face_colors" in mesh else np.full((len(triangles), 3), .6)
        if colors.size and colors.max() > 1:
            colors = colors / 255.0
        colors = np.clip(np.nan_to_num(colors, nan=.6), 0, 1)
    payload = {
        "title": blender.name, "mode": "Blender baseline only — model results pending",
        "constructionNote": construction_note(source_path, bi),
        "vertices": vertices.round(6).tolist(), "triangles": triangles.tolist(),
        "objectIds": ids.tolist(), "objectNames": names, "faceColors": colors.round(3).tolist(),
        "totalTriangles": total_triangles, "blenderCameras": [], "vggtCameras": [],
        "cloud": None,
        "totalCloudPoints": 0, "vggtSummary": None,
        "sourceWidth": manifest["width"], "sourceHeight": manifest["height"],
        "blenderWidth": bi["width"], "blenderHeight": bi["height"],
        "frames": [], "alignment": None, "manifestHash": source_hash,
        "landmarkProvenance": {"schema_version": 1, "source_manifest_sha256": source_hash,
                               "inputs": {"blender_index_sha256": sha256(blender / "index.json"),
                                          "blender_cameras_sha256": sha256(blender / "cameras.npz"),
                                          "blender_geometry_sha256": sha256(blender / "geometry.npz")}},
        "geometryNote": "Blender mesh is a single evaluated snapshot; moving objects do not follow the timeline here.",
    }
    for i in np.argsort(bc["frame_indices"]):
        index = bc["frame_indices"][i]
        payload["blenderCameras"].append({"index": int(index), "time": float(bc["time_seconds"][i]),
                                          "pose": bc["camera_to_world"][i].tolist(), "K": bc["intrinsics"][i].tolist()})
    if bool(args.vggt) != bool(args.alignment):
        raise ValueError("Supply --vggt and --alignment together for an aligned comparison")
    if args.vggt:
        from .alignment import load_vggt_ply, transform_cameras, validate_transform
        vggt = Path(args.vggt).resolve()
        ai_path = Path(args.alignment).resolve()
        ai = json.loads(ai_path.read_text())
        vi = json.loads((vggt / "index.json").read_text())
        if vi.get("schema_version") != 1 or vi.get("status") != "complete" or vi.get("source_manifest_sha256") != source_hash:
            raise ValueError("VGGT reconstruction must be complete and match the source manifest")
        if vi.get("preprocessing", {}).get("canonical_size_wh") != [manifest["width"], manifest["height"]]:
            raise ValueError("VGGT canonical camera dimensions differ from the source manifest")
        if (ai.get("schema_version") != 1 or ai.get("status") != "complete"
                or ai.get("source_manifest_sha256", ai.get("manifest_sha256")) != source_hash):
            raise ValueError("Alignment must be complete and match the source manifest")
        expected = {"vggt_index_sha256": sha256(vggt / "index.json"),
                    "blender_index_sha256": sha256(blender / "index.json"),
                    "vggt_cameras_sha256": sha256(vggt / "cameras.npz"),
                    "blender_cameras_sha256": sha256(blender / "cameras.npz")}
        if any(ai.get("inputs", {}).get(key) != value for key, value in expected.items()):
            raise ValueError("Alignment provenance differs from the loaded reconstruction/Blender export")
        expected["blender_geometry_sha256"] = payload["landmarkProvenance"]["inputs"]["blender_geometry_sha256"]
        expected["vggt_point_cloud_sha256"] = sha256(vggt / "point_cloud.ply")
        for key in ("blender_geometry_sha256", "vggt_point_cloud_sha256"):
            if key in ai["inputs"] and ai["inputs"][key] != expected[key]:
                raise ValueError(f"Alignment provenance {key} differs from the loaded geometry/cloud")
        payload["landmarkProvenance"]["inputs"].update(expected)
        transform = validate_transform(ai["transform"])
        cloud = load_vggt_ply(vggt / "point_cloud.ply", args.max_points)
        payload.update(cloud=pack_cloud(cloud, transform),
                       totalCloudPoints=cloud["total_points"],
                       alignment=ai, mode="VGGT aligned into Blender world with one global similarity transform")
        if ai.get("fit", {}).get("method") == "automatic_surfaces":
            status = ai["fit"].get("validation_status", "unverified").replace("_", " ")
            payload["mode"] = f"Automatic surface alignment · {status} · cameras evaluated after fitting"
        vc = _cameras(vggt / "cameras.npz", "intrinsics_canonical")
        _validate_camera_records(vc, vi, manifest)
        cloud_settings = vi.get("point_cloud", {})
        payload["vggtSummary"] = {
            "selectedViews": len(vc["frame_indices"]),
            "modelSizeWh": vi.get("preprocessing", {}).get("model_size_wh"),
            "confidencePercentile": cloud_settings.get("confidence_percentile_per_frame"),
            "pointStride": cloud_settings.get("point_stride"),
            "exportBudget": cloud_settings.get("max_points"),
        }
        poses = transform_cameras(vc["camera_to_world"], transform)
        for i in np.argsort(vc["frame_indices"]):
            index = vc["frame_indices"][i]
            payload["vggtCameras"].append({"index": int(index), "time": float(vc["time_seconds"][i]),
                                          "pose": poses[i].tolist(), "K": vc["intrinsics"][i].tolist()})
    max_images = max(0, int(args.max_images))
    records = {record["index"]: record for record in bi["frames"]}
    available_renders = sorted(record["index"] for record in bi["frames"] if record.get("rgb_file"))
    render_ids = {available_renders[i] for i in
                  np.linspace(0, len(available_renders) - 1, min(max_images, len(available_renders))).round().astype(int)}
    image_ids = set(render_ids)
    remaining = [frame["index"] for frame in manifest["frames"] if frame["index"] not in image_ids]
    slots = min(max_images - len(image_ids), len(remaining))
    image_ids.update(remaining[i] for i in np.linspace(0, len(remaining) - 1, slots).round().astype(int))
    for frame in manifest["frames"]:
        row = {"index": frame["index"], "time": frame["time_seconds"]}
        if frame["index"] in image_ids:
            row["image"] = _thumbnail(source_path.parent / frame["file"])
        record = records.get(frame["index"], {})
        if frame["index"] in render_ids:
            row["render"] = _thumbnail(blender / record["rgb_file"])
        payload["frames"].append(row)
    return payload


def run(args):
    output = Path(args.output).resolve()
    if output.suffix.lower() != ".html" or output.exists():
        raise ValueError("Viewer output must be a new .html file")
    if args.max_points < 1 or args.max_images < 0:
        raise ValueError("max_points must be positive and max_images nonnegative")
    payload = build_payload(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    before, after = HTML.split("__PAYLOAD__")
    # Stream the JSON to avoid a second complete HTML/base64 string in memory.
    with output.open("x") as stream:
        stream.write(before)
        encoder = json.JSONEncoder(allow_nan=False, separators=(",", ":"))
        for chunk in encoder.iterencode(payload):
            stream.write(chunk.replace("<", "\\u003c"))
        stream.write(after)
    print(f"Offline alignment viewer: {output}")
    return output


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scene alignment laboratory</title>
<style>
:root{color-scheme:dark;font:14px system-ui,-apple-system,sans-serif;background:#10151e;color:#e8edf5}*{box-sizing:border-box}body{margin:0;display:flex;flex-direction:column;height:100vh}header{flex-shrink:0;padding:18px 24px;border-bottom:1px solid #293342;display:flex;justify-content:space-between;gap:16px}h1{font-size:19px;margin:2px 0}header p{margin:4px 0;color:#a9b5c8;font-size:12px}.tag{color:#7de0d4;font-size:12px}main{display:grid;grid-template-columns:minmax(0,1fr) 300px;flex:1;min-height:0}.stage{position:relative;min-height:500px;background:radial-gradient(ellipse at 50% 35%,#243245,#0e1520 75%)}canvas{width:100%;height:100%;display:block;touch-action:none}.hint{position:absolute;bottom:15px;left:18px;max-width:80%;color:#aab9ce;font-size:11px;pointer-events:none}.status{position:absolute;top:16px;left:18px;background:#101722dd;border:1px solid #304258;border-radius:8px;padding:10px 12px;font:12px ui-monospace,monospace;white-space:pre-wrap}aside{overflow:auto;padding:18px;background:#151c27;border-left:1px solid #293342}h2{font-size:12px;text-transform:uppercase;letter-spacing:.1em;margin:20px 0 10px;color:#9bb1ca}h2:first-child{margin-top:0}label{display:block;margin:9px 0;color:#d2dce9;font-size:12px}input[type=checkbox]{accent-color:#62ccbf;margin-right:8px}input[type=range],select{width:100%}select,button{background:#202c3c;border:1px solid #3a4c63;color:#e8edf5;padding:7px;border-radius:5px;font:12px system-ui}button{cursor:pointer;margin:4px 4px 4px 0}button:hover{background:#314358}.note{font-size:11px;color:#9faec3;line-height:1.5}.images{display:flex;gap:6px}.images figure{width:50%;margin:0}.images img{width:100%;background:#0b1017;min-height:50px;object-fit:contain}.images figcaption{font-size:10px;color:#a4b4c9;margin-top:4px}.swatch{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:4px}.gold{background:#ffbd66}.purple{background:#b798ff}#pickInfo{font:11px ui-monospace,monospace;white-space:pre-wrap;overflow-wrap:anywhere}#error{display:none;color:#ffb6a8;background:#492725;padding:18px;position:absolute;inset:20px;z-index:3;height:max-content}@media(max-width:800px){main{display:block;height:auto}.stage{height:62vh}aside{border-left:0}.images img{max-height:220px}header{padding:12px}}
</style>
<header><div><div class="tag">RECONSTRUCTION / GEOMETRY</div><h1 id="title">Scene alignment laboratory</h1><p id="constructionNote" hidden></p><p id="mode"></p></div><div class="note">Offline · source frames preserved<br>Drag to orbit · wheel to zoom · shift-drag to pan</div></header>
<main><section class="stage"><canvas id="view"></canvas><div id="error"></div><div class="status" id="status"></div><div class="hint" id="meshNote"></div></section><aside>
<h2>Visible layers</h2><label><input id="wire" type="checkbox" checked>Blender wireframe</label><label><input id="solid" type="checkbox">Blender surfaces</label><label><input id="cloud" type="checkbox" checked>VGGT colored points</label><label><input id="cameras" type="checkbox" checked>Camera paths and selected frusta</label><label><input id="links" type="checkbox">Matched camera residual lines</label>
<label>Point size <input id="pointSize" type="range" min="1" max="6" step=".5" value="2"></label><label>Surface opacity <input id="opacity" type="range" min=".1" max="1" step=".05" value=".3"></label>
<label>Blender object <select id="object"><option value="all">All objects</option></select></label><button id="fit">Fit visible scene</button><button id="focus">Focus object</button>
<h2>Source timeline</h2><input id="frame" type="range" min="0" step="1" value="0"><div id="frameLabel" class="note"></div><div class="note"><span class="swatch purple"></span>Blender <span class="swatch gold"></span>VGGT</div><div class="images"><figure><img id="sourceImage" alt="Sampled original video frame"><figcaption id="sourceCaption"></figcaption></figure><figure><img id="renderImage" alt="Sampled Blender render"><figcaption id="renderCaption"></figcaption></figure></div>
<h2>Landmark pairs</h2><p class="note">Pick displayed vertices/points for a new global fit. Orbit to inspect each choice; clicks select the nearest projected candidate. The mesh is one static snapshot.</p><select id="pickMode"><option value="none">Orbit / pan</option><option value="source">Pick VGGT source point</option><option value="target">Pick Blender target vertex</option></select><div id="pickInfo">No points picked.</div><button id="addPair">Add pair</button><button id="undoPair">Undo pair</button><button id="download">Download landmarks</button><p class="note" id="pairCount">0 pairs. Use at least 3 non-collinear points, preferably more.</p>
<h2>Interpretation</h2><p class="note">Point counts below distinguish the displayed cloud from the export. The Blender mesh is sampled for display. Camera fitting can hide a global pose offset; inspect static surfaces and held-out frames too. Neither model output nor a close overlay proves the scene is correct.</p><p class="note" id="stats"></p><p class="note" id="cloudDetails" style="white-space:pre-line"></p>
</aside></main><script id="data" type="application/json">__PAYLOAD__</script><script>
'use strict';
const dataElement=document.getElementById('data'),D=JSON.parse(dataElement.textContent),$=id=>document.getElementById(id);
dataElement.textContent='';dataElement.remove();
// Decode bounded base64 chunks; release encoded strings after creating typed arrays.
function decodeBytes(encoded,size){const result=new Uint8Array(size);let offset=0;for(let start=0;start<encoded.length;start+=1048576){const part=atob(encoded.slice(start,start+1048576));for(let i=0;i<part.length;i++)result[offset++]=part.charCodeAt(i)}if(offset!==size)throw Error('Invalid point-cloud byte length');return result}
function decodeTyped(encoded,count,Type){const bytes=decodeBytes(encoded,count*Type.BYTES_PER_ELEMENT);if(new Uint8Array(new Uint32Array([1]).buffer)[0]!==1){const width=Type.BYTES_PER_ELEMENT;for(let i=0;i<bytes.length;i+=width)for(let j=0;j<width/2;j++){const tmp=bytes[i+j];bytes[i+j]=bytes[i+width-1-j];bytes[i+width-1-j]=tmp}}return new Type(bytes.buffer)}
const cloud=D.cloud,pointCount=cloud?cloud.count:0;
let cloudPositions=new Float32Array(0),cloudColors=new Uint8Array(0),cloudFrames=new Int32Array(0),cloudMatrix=null;
if(cloud){cloudPositions=decodeTyped(cloud.positionsF32,pointCount*3,Float32Array);delete cloud.positionsF32;cloudColors=decodeBytes(cloud.colorsU8,pointCount*3);delete cloud.colorsU8;cloudFrames=decodeTyped(cloud.frameIndicesI32,pointCount,Int32Array);delete cloud.frameIndicesI32;cloudMatrix=new Float32Array(cloud.matrix)}
$('title').textContent=D.title+' · alignment laboratory';$('constructionNote').textContent=D.constructionNote||'';$('constructionNote').hidden=!D.constructionNote;$('mode').textContent=D.mode;$('meshNote').textContent=D.geometryNote;
$('frame').max=D.frames.length-1;$('cloud').disabled=!pointCount;$('pickMode').options[1].disabled=!pointCount;
$('stats').textContent=`${D.triangles.length.toLocaleString()} / ${D.totalTriangles.toLocaleString()} triangles · ${D.blenderCameras.length} Blender cameras · ${D.vggtCameras.length} VGGT cameras`;
if(D.vggtSummary){const s=D.vggtSummary,total=D.totalCloudPoints,percentage=total?100*pointCount/total:0,lines=[`Display: ${pointCount.toLocaleString()} / ${total.toLocaleString()} exported points (${percentage.toFixed(1)}%).`,`Inference: ${s.selectedViews.toLocaleString()} views${s.modelSizeWh?' at '+s.modelSizeWh.join(' × ')+' pixels per view':''}.`];if(s.confidencePercentile!==null)lines.push(`Cloud export: per-frame confidence percentile cutoff ${s.confidencePercentile}; confidence is not a probability.`);if(s.pointStride!==null||s.exportBudget!==null)lines.push(`Export sampling: ${[s.pointStride!==null?'pixel stride '+s.pointStride:null,s.exportBudget!==null?'point cap '+s.exportBudget.toLocaleString():null].filter(Boolean).join(', ')}.`);lines.push('Display sampling does not change saved depth or cameras.');$('cloudDetails').textContent=lines.join('\n')}
D.objectNames.forEach((name,i)=>{const o=document.createElement('option');o.value=i;o.textContent=name;$('object').appendChild(o)});
const V=(a,b)=>a.map((v,i)=>v-b[i]),add=(a,b)=>a.map((v,i)=>v+b[i]),mul=(a,s)=>a.map(v=>v*s),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0),cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]],norm=a=>mul(a,1/(Math.hypot(...a)||1));
function mm(a,b){const c=new Float32Array(16);for(let col=0;col<4;col++)for(let row=0;row<4;row++)for(let k=0;k<4;k++)c[col*4+row]+=a[k*4+row]*b[col*4+k];return c}
function look(eye,at){const z=norm(V(eye,at)),x=norm(cross([0,0,1],z)),y=cross(z,x);return new Float32Array([x[0],y[0],z[0],0,x[1],y[1],z[1],0,x[2],y[2],z[2],0,-dot(x,eye),-dot(y,eye),-dot(z,eye),1])}
function perspective(aspect,near,far){const f=1/Math.tan(55*Math.PI/360);return new Float32Array([f/aspect,0,0,0,0,f,0,0,0,0,(far+near)/(near-far),-1,0,0,2*far*near/(near-far),0])}
const canvas=$('view'),gl=canvas.getContext('webgl',{antialias:true,alpha:false});
let center=[0,0,0],radius=8,yaw=-1.15,pitch=.42,VP,meshLines,meshSolid,cloudBuffer,pathsBuffer,linkBuffer,selectedVertices=D.vertices;
function fail(message){$('error').style.display='block';$('error').textContent=message;throw Error(message)}
if(!gl)fail('WebGL is unavailable in this browser. Open this file in a browser with hardware acceleration.');
function shader(type,code){const s=gl.createShader(type);gl.shaderSource(s,code);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))fail(gl.getShaderInfoLog(s));return s}
const program=gl.createProgram();gl.attachShader(program,shader(gl.VERTEX_SHADER,'attribute vec3 aPosition;attribute vec3 aColor;uniform mat4 uVP;uniform float uSize;varying vec3 vColor;void main(){gl_Position=uVP*vec4(aPosition,1.);gl_PointSize=uSize;vColor=aColor;}'));gl.attachShader(program,shader(gl.FRAGMENT_SHADER,'precision mediump float;varying vec3 vColor;uniform float uAlpha;uniform bool uPoints;void main(){if(uPoints && distance(gl_PointCoord,vec2(.5))>.5)discard;gl_FragColor=vec4(vColor,uAlpha);}'));gl.linkProgram(program);if(!gl.getProgramParameter(program,gl.LINK_STATUS))fail(gl.getProgramInfoLog(program));gl.useProgram(program);
const loc={p:gl.getAttribLocation(program,'aPosition'),c:gl.getAttribLocation(program,'aColor'),vp:gl.getUniformLocation(program,'uVP'),size:gl.getUniformLocation(program,'uSize'),alpha:gl.getUniformLocation(program,'uAlpha'),points:gl.getUniformLocation(program,'uPoints')};
function buffer(points,colors){const a=new Float32Array(points.length*6);points.forEach((p,i)=>{a.set(p,i*6);a.set(colors[i]||[.7,.7,.7],i*6+3)});const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,a,gl.STATIC_DRAW);return {b,count:points.length}}
function cloudVertexBuffer(){if(!pointCount)return null;const b=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,b);gl.bufferData(gl.ARRAY_BUFFER,cloudPositions,gl.STATIC_DRAW);const colorBuffer=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,colorBuffer);gl.bufferData(gl.ARRAY_BUFFER,cloudColors,gl.STATIC_DRAW);cloudColors=null;return {b,colorBuffer,count:pointCount}}
function draw(b,mode,alpha=1,size=2,matrix=VP){if(!b||!b.count)return;gl.uniformMatrix4fv(loc.vp,false,matrix);gl.bindBuffer(gl.ARRAY_BUFFER,b.b);gl.enableVertexAttribArray(loc.p);gl.enableVertexAttribArray(loc.c);if(b.colorBuffer){gl.vertexAttribPointer(loc.p,3,gl.FLOAT,false,0,0);gl.bindBuffer(gl.ARRAY_BUFFER,b.colorBuffer);gl.vertexAttribPointer(loc.c,3,gl.UNSIGNED_BYTE,true,0,0)}else{gl.vertexAttribPointer(loc.p,3,gl.FLOAT,false,24,0);gl.vertexAttribPointer(loc.c,3,gl.FLOAT,false,24,12)}gl.uniform1f(loc.alpha,alpha);gl.uniform1f(loc.size,size);gl.uniform1i(loc.points,mode===gl.POINTS);gl.drawArrays(mode,0,b.count)}
const bc=new Map(D.blenderCameras.map(c=>[c.index,c])),vc=new Map(D.vggtCameras.map(c=>[c.index,c]));
function origin(c){return c.pose.slice(0,3).map(row=>row[3])}
function rebuildMesh(){if(meshLines)gl.deleteBuffer(meshLines.b);if(meshSolid)gl.deleteBuffer(meshSolid.b);const chosen=$('object').value,p=[],pc=[],l=[],lc=[],used=new Set();D.triangles.forEach((t,i)=>{if(chosen!=='all'&&D.objectIds[i]!==+chosen)return;const v=t.map(j=>D.vertices[j]),base=D.faceColors[i];t.forEach(j=>used.add(j));p.push(...v);pc.push(base,base,base);for(const [a,b] of [[0,1],[1,2],[2,0]]){l.push(v[a],v[b]);lc.push([.35,.65,.68],[.35,.65,.68])}});selectedVertices=[...used].map(i=>D.vertices[i]);meshLines=buffer(l,lc);meshSolid=buffer(p,pc)}
function cameraPaths(){const p=[],c=[],r=[],rc=[];for(const [list,col] of [[D.blenderCameras,[.7,.59,1]],[D.vggtCameras,[1,.74,.4]]])for(let i=1;i<list.length;i++){p.push(origin(list[i-1]),origin(list[i]));c.push(col,col)}for(const cam of D.vggtCameras){if(bc.has(cam.index)){r.push(origin(cam),origin(bc.get(cam.index)));rc.push([.95,.4,.4],[.95,.4,.4])}}pathsBuffer=buffer(p,c);linkBuffer=buffer(r,rc)}
function frustum(cam,width,height,col){if(!cam)return null;const scale=radius*.045,K=cam.K,C=cam.pose,o=origin(cam),p=[],colors=[];const corner=(u,v)=>{const y=(v-K[1][2])/K[1][1],x=(u-K[0][2]-K[0][1]*y)/K[0][0],ray=[x,y,1];return C.slice(0,3).map(row=>row[3]+scale*dot(row.slice(0,3),ray))};const q=[[-.5,-.5],[width-.5,-.5],[width-.5,height-.5],[-.5,height-.5]].map(([u,v])=>corner(u,v));for(let i=0;i<4;i++){p.push(o,q[i],q[i],q[(i+1)%4]);colors.push(col,col,col,col)}return buffer(p,colors)}
function fit(points){if(!points.length)return;const lo=[Infinity,Infinity,Infinity],hi=[-Infinity,-Infinity,-Infinity];points.forEach(p=>p.forEach((v,i)=>{lo[i]=Math.min(lo[i],v);hi[i]=Math.max(hi[i],v)}));center=lo.map((v,i)=>(v+hi[i])/2);radius=Math.max(.1,Math.hypot(...V(hi,lo))*.9);render()}
function frameInfo(){const i=+$('frame').value,frame=D.frames[i];$('frameLabel').textContent=`Frame ${String(frame.index).padStart(6,'0')} · ${frame.time.toFixed(3)} s`;function show(key,img,caption,label){const available=D.frames.filter(f=>f[key]);if(!available.length){$(img).removeAttribute('src');$(caption).textContent=label+' unavailable';return}const closest=available.reduce((a,b)=>Math.abs(a.index-frame.index)<=Math.abs(b.index-frame.index)?a:b);$(img).src=closest[key];$(caption).textContent=`${label}: frame ${closest.index}${closest.index===frame.index?'':' (nearest preview)'}`};show('image','sourceImage','sourceCaption','Video');show('render','renderImage','renderCaption','Blender');const b=bc.get(frame.index),v=vc.get(frame.index);let text=`Frame ${frame.index}  |  ${frame.time.toFixed(3)} s`;if(b&&v){const distance=Math.hypot(...V(origin(b),origin(v)));let tr=0;for(let row=0;row<3;row++)for(let k=0;k<3;k++)tr+=b.pose[row][k]*v.pose[row][k];const angle=Math.acos(Math.max(-1,Math.min(1,(tr-1)/2)))*180/Math.PI;text+=`\nCamera separation: ${distance.toFixed(3)} Blender units\nOrientation difference: ${angle.toFixed(2)}°`}else text+=v?'':'\nNo VGGT camera on this source frame';$('status').textContent=text}
function render(){const ratio=Math.min(window.devicePixelRatio||1,2),w=Math.max(1,canvas.clientWidth),h=Math.max(1,canvas.clientHeight);if(canvas.width!==Math.round(w*ratio)||canvas.height!==Math.round(h*ratio)){canvas.width=Math.round(w*ratio);canvas.height=Math.round(h*ratio)}gl.viewport(0,0,canvas.width,canvas.height);gl.clearColor(.055,.08,.12,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);const eye=add(center,[radius*Math.cos(pitch)*Math.cos(yaw),radius*Math.cos(pitch)*Math.sin(yaw),radius*Math.sin(pitch)]);VP=mm(perspective(w/h,radius/2000,radius*100),look(eye,center));gl.uniformMatrix4fv(loc.vp,false,VP);if($('solid').checked){gl.depthMask(false);draw(meshSolid,gl.TRIANGLES,+$('opacity').value);gl.depthMask(true)}if($('wire').checked)draw(meshLines,gl.LINES,.45);if($('cloud').checked)draw(cloudBuffer,gl.POINTS,1,+$('pointSize').value*ratio,cloudMatrix?mm(VP,cloudMatrix):VP);if($('cameras').checked){gl.disable(gl.DEPTH_TEST);draw(pathsBuffer,gl.LINES);const idx=D.frames[+$('frame').value].index;for(const [cam,width,height,col] of [[bc.get(idx),D.blenderWidth,D.blenderHeight,[.7,.59,1]],[vc.get(idx),D.sourceWidth,D.sourceHeight,[1,.74,.4]]]){const f=frustum(cam,width,height,col);draw(f,gl.LINES);if(f)gl.deleteBuffer(f.b)}}if($('links').checked)draw(linkBuffer,gl.LINES,.7)}
function project(p){const v=[...p,1],q=[0,0,0,0];for(let row=0;row<4;row++)for(let k=0;k<4;k++)q[row]+=VP[k*4+row]*v[k];if(q[3]<=0)return null;return [(q[0]/q[3]*.5+.5)*canvas.clientWidth,(-q[1]/q[3]*.5+.5)*canvas.clientHeight,q[2]/q[3]]}
let pending={source:null,target:null},pairs=[];
function pick(x,y){const kind=$('pickMode').value;if(kind==='none')return;const source=kind==='source',count=source?pointCount:selectedVertices.length,m=source&&cloudMatrix?mm(VP,cloudMatrix):VP,w=canvas.clientWidth,h=canvas.clientHeight;let best=-1,dist=14*14,bestZ=Infinity;for(let i=0;i<count;i++){const px=source?cloudPositions[i*3]:selectedVertices[i][0],py=source?cloudPositions[i*3+1]:selectedVertices[i][1],pz=source?cloudPositions[i*3+2]:selectedVertices[i][2],qw=m[3]*px+m[7]*py+m[11]*pz+m[15];if(qw<=0)continue;const qz=(m[2]*px+m[6]*py+m[10]*pz+m[14])/qw;if(qz< -1||qz>1)continue;const qx=((m[0]*px+m[4]*py+m[8]*pz+m[12])/qw*.5+.5)*w,qy=(-(m[1]*px+m[5]*py+m[9]*pz+m[13])/qw*.5+.5)*h,d=(qx-x)**2+(qy-y)**2;if(d<dist||(Math.abs(d-dist)<.3&&qz<bestZ)){best=i;dist=d;bestZ=qz}}if(best<0){$('pickInfo').textContent='No displayed candidate within 14 pixels. Zoom in or change view.';return}pending[kind]=source?Array.from(cloudPositions.subarray(best*3,best*3+3)):selectedVertices[best];$('pickInfo').textContent=JSON.stringify(pending,null,2)}
let drag=null;canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,startX:e.clientX,startY:e.clientY,pan:e.shiftKey};canvas.setPointerCapture(e.pointerId)});canvas.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(drag.pan){const right=[-Math.sin(yaw),Math.cos(yaw),0],up=[-Math.sin(pitch)*Math.cos(yaw),-Math.sin(pitch)*Math.sin(yaw),Math.cos(pitch)];center=add(center,add(mul(right,-dx*radius/500),mul(up,dy*radius/500)))}else{yaw-=dx*.008;pitch=Math.max(-1.5,Math.min(1.5,pitch+dy*.008))}drag.x=e.clientX;drag.y=e.clientY;render()});canvas.addEventListener('pointerup',e=>{if(drag&&Math.hypot(e.clientX-drag.startX,e.clientY-drag.startY)<4){const r=canvas.getBoundingClientRect();pick(e.clientX-r.left,e.clientY-r.top)}drag=null});canvas.addEventListener('wheel',e=>{e.preventDefault();radius=Math.max(.01,Math.min(1e6,radius*Math.exp(e.deltaY*.001)));render()},{passive:false});
for(const id of ['wire','solid','cloud','cameras','links','pointSize','opacity'])$(id).addEventListener('input',render);$('frame').addEventListener('input',()=>{frameInfo();render()});$('object').addEventListener('change',()=>{rebuildMesh();render()});$('focus').onclick=()=>fit(selectedVertices);$('fit').onclick=()=>fit(D.vertices.concat(cloud&&cloud.bounds?cloud.bounds:[],D.blenderCameras.map(origin)));$('addPair').onclick=()=>{if(!pending.source||!pending.target){$('pickInfo').textContent='Pick one VGGT point and one Blender vertex first.';return}pairs.push({name:'landmark_'+String(pairs.length+1).padStart(2,'0'),source:pending.source,target:pending.target,use_for_fit:true});pending={source:null,target:null};$('pickInfo').textContent='Pair saved. Pick the next correspondence.';$('pairCount').textContent=`${pairs.length} pairs saved. Review correspondences before fitting.`};$('undoPair').onclick=()=>{pairs.pop();$('pairCount').textContent=`${pairs.length} pairs saved.`};$('download').onclick=()=>{const blob=new Blob([JSON.stringify({...D.landmarkProvenance,pairs},null,2)],{type:'application/json'}),a=document.createElement('a'),url=URL.createObjectURL(blob);a.href=url;a.download='landmarks.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
cloudBuffer=cloudVertexBuffer();cameraPaths();rebuildMesh();frameInfo();fit(D.vertices.concat(D.blenderCameras.map(origin)));new ResizeObserver(render).observe(canvas);
</script></html>'''
