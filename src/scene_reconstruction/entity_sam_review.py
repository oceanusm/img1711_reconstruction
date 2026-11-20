"""Inspect unreviewed SAM concept unions against visible Blender entity unions.

These diagnostics deliberately do not assign SAM tracks to physical objects.
Metrics are computed on the original pixel grid; small browser previews are
display assets only. The HTML embeds its data and uses ordinary image layers,
so it can be opened from disk without a server or cross-origin canvas reads.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from .common import sha256, write_json
from .compare import read_mask
from .entity_artifacts import load_bundle, load_labels
from .instances import load_sam_run


SCENES = (("before_vqa", "Original before VQA", "entities_1711_before_vqa.json"),
          ("vggt_guided", "VGGT guided", "entities_1711_vggt_guided.json"))
INTERPRETATION = (
    "Unreviewed concept-union diagnostics, not individual-object scores. "
    "Blender entities sharing a concept are unioned; SAM may detect additional real items. "
    "SAM is an estimated reference, not verified ground truth. SAM-empty frames do not "
    "establish object absence. Different scene inventories and segmentation errors can "
    "change these scores; they are not a verdict on which reconstruction is better."
)


def concept_inventory(entities):
    """Return whole-entity groupings; duplicate concept text must agree."""
    concepts = {}
    for entity in entities:
        concept = entity["concept"]
        group = concepts.setdefault(concept, {"text": entity["sam_text"], "entities": [], "ids": []})
        if group["text"] != entity["sam_text"]:
            raise ValueError(f"Conflicting SAM texts for concept {concept}")
        group["entities"].append(entity["name"])
        group["ids"].append(entity["id"])
    return concepts


def concept_label_image(labels, inventory):
    """Relabel visible entity IDs once; several entities may share one concept."""
    ids = [entity_id for group in inventory.values() for entity_id in group["ids"]]
    if not ids or len(ids) != len(set(ids)) or min(ids) < 1:
        raise ValueError("Concept groups must assign each positive entity ID once")
    if labels.ndim != 2 or not np.isin(labels, [0, *ids]).all():
        raise ValueError("Labels contain an unknown entity ID or invalid grid")
    names = sorted(inventory)
    lookup = np.zeros(max(ids) + 1, dtype=np.uint16)
    for ordinal, name in enumerate(names, 1):
        lookup[inventory[name]["ids"]] = ordinal
    return lookup[labels], {name: ordinal for ordinal, name in enumerate(names, 1)}


def overlap_metrics(blender, sam):
    """Blender is the prediction; SAM is an explicitly unreviewed reference."""
    if blender.shape != sam.shape or blender.ndim != 2:
        raise ValueError("Masks must use the same two-dimensional pixel grid")
    blender, sam = blender.astype(bool, copy=False), sam.astype(bool, copy=False)
    b_area, s_area = int(np.count_nonzero(blender)), int(np.count_nonzero(sam))
    intersection = int(np.count_nonzero(blender & sam))
    union = b_area + s_area - intersection
    presence = ("both_nonempty" if b_area and s_area else "sam_empty_unresolved" if b_area
                else "blender_empty_sam_nonempty" if s_area else "both_empty_no_overlap_evidence")
    return {"intersection": intersection, "union": union, "blender_area": b_area,
            "sam_area": s_area, "iou": intersection / union if union else None,
            "dice": 2 * intersection / (b_area + s_area) if b_area + s_area else None,
            "precision_vs_sam": intersection / b_area if b_area else None,
            "recall_vs_sam": intersection / s_area if s_area else None,
            "presence": presence, "both_empty": union == 0}


def summarize(rows):
    intersections = sum(row["intersection"] for row in rows)
    union = sum(row["union"] for row in rows)
    blender = sum(row["blender_area"] for row in rows)
    sam = sum(row["sam_area"] for row in rows)
    ious = [row["iou"] for row in rows if row["iou"] is not None]
    return {"frame_count": len(rows), "frames_with_nonempty_union": len(ious),
            "both_empty_frames": sum(row["both_empty"] for row in rows),
            "sam_empty_blender_nonempty_frames": sum(row["presence"] == "sam_empty_unresolved" for row in rows),
            "blender_empty_sam_nonempty_frames": sum(row["presence"] == "blender_empty_sam_nonempty" for row in rows),
            "mean_frame_iou": sum(ious) / len(ious) if ious else None,
            "pixel_pooled_iou": intersections / union if union else None,
            "pixel_pooled_dice": 2 * intersections / (blender + sam) if blender + sam else None,
            "precision_vs_sam": intersections / blender if blender else None,
            "recall_vs_sam": intersections / sam if sam else None,
            "intersection": intersections, "union": union, "blender_area": blender, "sam_area": sam}


def shared_references(scenes):
    """Compare decoded masks, not compressed PNG bytes or SAM instance IDs."""
    if len(scenes) != 2:
        raise ValueError("Paired reference reporting requires exactly two scenes")
    first, second = scenes
    result = {}
    for name in sorted(set(first["concepts"]) & set(second["concepts"])):
        a, b = first["concepts"][name], second["concepts"][name]
        a_rows, b_rows = a["frames"], b["frames"]
        if [r["index"] for r in a_rows] != [r["index"] for r in b_rows]:
            raise ValueError("Shared concepts have different source frame IDs")
        differing = [x["index"] for x, y in zip(a_rows, b_rows)
                     if x["sam_pixel_sha256"] != y["sam_pixel_sha256"]]
        same_prompt = a["text"] == b["text"] and a["anchor_frame"] == b["anchor_frame"]
        result[name] = {"same_text_and_anchor": same_prompt, "identical_masks": not differing,
                        "differing_frame_ids": differing, "differing_frame_count": len(differing),
                        "reference_comparable": same_prompt and not differing,
                        "inventory_sizes": [len(a["entities"]), len(b["entities"])]}
    return result


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Input artifact path escapes its stage directory")
    return path


def _read_scene(root, key, label, config):
    directory = root / "outputs/entity_comparison_1711" / key
    manifest_path = directory / "sam_inputs/manifest.json"
    index_path = directory / "render/index.json"
    entities_path = root / "configs" / config
    index, entities, manifest, originals = load_bundle(index_path, entities_path, manifest_path, sparse=True)
    sam_manifest, metadata, _, _ = load_sam_run(manifest_path, directory / "sam")
    inventory = concept_inventory(entities["entities"])
    prompts = {item["name"]: item for item in metadata["objects"]}
    if set(inventory) != set(prompts):
        raise ValueError(f"{key}: SAM concepts differ from entity inventory")
    for concept, group in inventory.items():
        if group["text"] != prompts[concept]["text"]:
            raise ValueError(f"{key}: SAM text differs from entity inventory: {concept}")
    records = {}
    with (directory / "sam/frames.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            records[(row["object"], row["index"])] = row
    return {"key": key, "label": label, "directory": directory, "manifest_path": manifest_path,
            "index": index, "manifest": sam_manifest, "originals": originals, "inventory": inventory,
            "prompts": prompts, "records": records, "provenance": {
                "source_manifest_sha256": index["source_manifest_sha256"],
                "selection_sha256": index["selection_sha256"], "entities_sha256": sha256(entities_path),
                "render_index_sha256": sha256(index_path), "sam_manifest_sha256": sha256(manifest_path),
                "sam_metadata_sha256": sha256(directory / "sam/metadata.json"),
                "sam_frames_sha256": sha256(directory / "sam/frames.jsonl")}}


def _validate_pair(first, second):
    for field in ("width", "height", "source_manifest_sha256", "selection_sha256"):
        if first["index"][field] != second["index"][field]:
            raise ValueError(f"Paired scenes disagree on {field}")
    if list(first["originals"]) != list(second["originals"]):
        raise ValueError("Paired scenes have different selected source frames")
    for index, frame in first["originals"].items():
        other = second["originals"][index]
        if any(frame[key] != other[key] for key in ("time_seconds", "pts_seconds")):
            raise ValueError("Paired scene source timestamps differ")
        if ((first["manifest_path"].parent / frame["file"]).resolve() !=
                (second["manifest_path"].parent / other["file"]).resolve()):
            raise ValueError("Paired scenes reference different source RGB files")


def _rgb_preview(path, destination, expected_size, preview_size):
    with Image.open(path) as image:
        if image.size != expected_size:
            raise ValueError(f"RGB dimensions differ from mask grid: {path}")
        image.convert("RGB").resize(preview_size, Image.Resampling.LANCZOS).save(destination, quality=88)


def _sprite(blender, sam, size, destination):
    """One transparent three-panel sprite: SAM, Blender, and their difference."""
    b = np.asarray(Image.fromarray(blender).resize(size, Image.Resampling.NEAREST), dtype=bool)
    s = np.asarray(Image.fromarray(sam).resize(size, Image.Resampling.NEAREST), dtype=bool)
    width, height = size
    sprite = np.zeros((height, width * 3, 4), dtype=np.uint8)
    sprite[:, :width][s] = [232, 70, 209, 255]
    sprite[:, width:2*width][b] = [40, 215, 231, 255]
    difference = sprite[:, 2*width:]
    difference[s & ~b] = [232, 70, 209, 255]
    difference[b & ~s] = [40, 215, 231, 255]
    difference[b & s] = [255, 255, 255, 255]
    Image.fromarray(sprite).save(destination, compress_level=3)


def _review_notes(root, output):
    path = root / "outputs/entity_comparison_1711/identity_review/candidate_notes.json"
    if not path.is_file():
        return {"notes": [], "review_url": None, "status": "not_available"}
    notes = json.loads(path.read_text())
    return {"status": notes.get("status"), "notes": notes.get("notes", []),
            "source_sha256": sha256(path),
            "scope": "Sampled visual review only; not an approved physical-instance mapping",
            "review_url": Path(os.path.relpath(path.parent / "index.html", output)).as_posix()}


def run(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    height = args.preview_height
    if not isinstance(height, int) or height < 32:
        raise ValueError("--preview-height must be an integer >=32")
    scenes = [_read_scene(root, *spec) for spec in SCENES]
    _validate_pair(*scenes)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output must be new or empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    size = scenes[0]["index"]["width"], scenes[0]["index"]["height"]
    height = min(height, size[1])
    preview_size = max(1, round(size[0] * height / size[1])), height
    frames = [{"index": item["index"], "time_seconds": item["time_seconds"],
               "pts_seconds": item["pts_seconds"], "rgb": f"rgb/source/{item['index']:06d}.jpg"}
              for item in scenes[0]["manifest"]["frames"]]
    report = {"schema_version": 1, "stage": "unreviewed_concept_union_diagnostics", "status": "running",
              "interpretation": INTERPRETATION, "width": size[0], "height": size[1],
              "preview_width": preview_size[0], "preview_height": preview_size[1], "frames": frames,
              "metric_convention": "Blender prediction versus unreviewed SAM reference; both-empty IoU/Dice undefined",
              "review_notes": _review_notes(root, output), "scenes": []}
    write_json(output / "metadata.json", {k: v for k, v in report.items() if k not in ("frames", "scenes")})
    csv_rows = []
    try:
        (output / "rgb/source").mkdir(parents=True)
        for frame in frames:
            original = scenes[0]["originals"][frame["index"]]
            _rgb_preview(scenes[0]["manifest_path"].parent / original["file"], output / frame["rgb"], size, preview_size)
        for scene in scenes:
            key = scene["key"]
            (output / f"rgb/{key}").mkdir(parents=True)
            result = {"key": key, "label": scene["label"], "provenance": scene["provenance"],
                      "concepts": {}, "concept_count": len(scene["inventory"]),
                      "entity_count": sum(len(g["ids"]) for g in scene["inventory"].values()),
                      "catalog": Path(os.path.relpath(scene["directory"] / "sam_catalog/index.html", output)).as_posix()}
            for name, group in sorted(scene["inventory"].items()):
                (output / f"masks/{key}/{name}").mkdir(parents=True)
                result["concepts"][name] = {"text": group["text"], "entities": group["entities"],
                    "anchor_frame": scene["prompts"][name]["frame_index"], "frames": []}
            for position, frame in enumerate(scene["index"]["frames"]):
                frame_id = frame["index"]
                labels = load_labels(_inside(scene["directory"] / "render", frame["entity_file"]),
                                     (size[1], size[0]), [i for g in scene["inventory"].values() for i in g["ids"]])
                concept_labels, ordinals = concept_label_image(labels, scene["inventory"])
                _rgb_preview(_inside(scene["directory"] / "render", frame["rgb_file"]),
                             output / f"rgb/{key}/{frame_id:06d}.jpg", size, preview_size)
                for name, ordinal in ordinals.items():
                    record = scene["records"][(name, frame_id)]
                    sam = read_mask(_inside(scene["directory"] / "sam", record["mask"]), size)
                    if int(np.count_nonzero(sam)) != record["area_pixels"]:
                        raise ValueError(f"SAM mask area differs from metadata: {key}/{name}/{frame_id}")
                    blender = concept_labels == ordinal
                    metrics = overlap_metrics(blender, sam)
                    sprite = f"masks/{key}/{name}/{frame_id:06d}.png"
                    _sprite(blender, sam, preview_size, output / sprite)
                    row = {"index": frame_id, "time_seconds": frame["time_seconds"], **metrics,
                           "sam_instances": len(record["instances"]), "sprite": sprite,
                           "sam_pixel_sha256": hashlib.sha256(np.packbits(sam).tobytes()).hexdigest()}
                    result["concepts"][name]["frames"].append(row)
                    csv_rows.append({"scene": key, "concept": name,
                                     **{k: v for k, v in row.items() if k not in ("sprite", "sam_pixel_sha256")}})
                if position % 10 == 0 or position + 1 == len(frames):
                    print(f"Concept review {key}: {position+1}/{len(frames)} frames", flush=True)
            for concept in result["concepts"].values():
                concept["summary"] = summarize(concept["frames"])
            result["zero_detection_concepts"] = [name for name, c in result["concepts"].items()
                                                 if c["summary"]["sam_area"] == 0]
            result["summary"] = summarize([row for c in result["concepts"].values() for row in c["frames"]])
            result["summary"]["scope"] = "All scene concept-frame pairs; inventory-dependent diagnostic only"
            report["scenes"].append(result)
        report["shared_references"] = shared_references(report["scenes"])
        report["shared_concept_count"] = len(report["shared_references"])
        report["status"] = "complete"
        with (output / "per_frame.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
            writer.writeheader()
            writer.writerows(csv_rows)
        with (output / "per_concept.csv").open("w", newline="") as handle:
            summaries = [{"scene": s["key"], "concept": name, "entity_count": len(c["entities"]),
                          **c["summary"]} for s in report["scenes"] for name, c in s["concepts"].items()]
            writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
            writer.writeheader()
            writer.writerows(summaries)
        write_json(output / "review.json", report)
        (output / "index.html").write_text(_html(report))
        write_json(output / "metadata.json", {k: v for k, v in report.items() if k not in ("frames", "scenes")})
    except BaseException as error:
        write_json(output / "metadata.json", {"schema_version": 1, "status": "failed", "error": f"{type(error).__name__}: {error}"})
        raise
    print(f"Concept review ready: {output / 'index.html'}")


def _html(report):
    payload = json.dumps(report, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    return HTML.replace("__REPORT__", payload)


HTML = r'''<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Kitchen 1711 · SAM and Blender review</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#101319;color:#e9edf3;--opacity:.56}
*{box-sizing:border-box}body{margin:0 auto;max-width:1480px;padding:26px}h1{font-size:27px;margin:0 0 9px}h2{font-size:19px;margin:0}p{line-height:1.5}.muted{color:#aeb8c8}.warning{padding:12px 16px;border-left:3px solid #edbd67;background:#25221b;font-size:14px;max-width:1100px}
a{color:#85c7ff}button,select,input{font:inherit}button,select{background:#222a36;border:1px solid #4e5a6d;color:#fff;border-radius:6px;padding:8px 11px}button{cursor:pointer}button:disabled{opacity:.4;cursor:default}label{display:flex;gap:7px;align-items:center}.controls{display:flex;flex-wrap:wrap;align-items:center;gap:14px;margin:20px 0 12px}.controls select{max-width:360px}.timeline{display:grid;grid-template-columns:auto 1fr auto;align-items:center;gap:14px}#position{width:100%;accent-color:#87c4f3}#stamp{font-variant-numeric:tabular-nums;min-width:205px}.legend{display:flex;gap:20px;margin:14px 0;font-size:14px;flex-wrap:wrap}.dot{width:12px;height:12px;display:inline-block;border-radius:2px;margin-right:6px}.sam{background:#e846d1}.blender{background:#28d7e7}.overlap{background:#fff}.scene{border:1px solid #364152;border-radius:10px;margin:18px 0;overflow:hidden;background:#171d26}.scenehead{padding:14px 18px;display:flex;flex-wrap:wrap;justify-content:space-between;gap:10px}.stats{font-size:14px;line-height:1.7;font-variant-numeric:tabular-nums}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:2px;background:#303845}.panel{background:#121720;padding:10px}.panel h3{font-size:13px;font-weight:500;margin:0 0 9px;color:#c9d4e4}.image{position:relative;width:100%;max-width:267.1875px;aspect-ratio:9/16;margin:auto;overflow:hidden}.image img{width:100%;height:100%;display:block;object-fit:fill}.mask{position:absolute;inset:0;background-size:300% 100%;background-repeat:no-repeat;opacity:var(--opacity)}.mask.samLayer{background-position:0% 0}.mask.blenderLayer{background-position:50% 0}.mask.diffLayer{background-position:100% 0}.details{padding:11px 18px;font-size:13px;color:#b5c2d6}.missing{padding:25px;color:#edbd67}.ref{font-size:14px;margin:8px 0;color:#b6c4d7}#trend{height:42px;width:100%;display:flex;gap:1px;align-items:flex-end;position:relative;margin:8px 0}.bar{flex:1;min-width:1px;position:relative;background:#2a3140;height:100%;border:0;padding:0;border-radius:0}.bar span{position:absolute;bottom:0;width:47%;background:#e9a567;left:0;min-height:1px}.bar span:nth-child(2){left:53%;background:#78b6f1}.bar.current{outline:2px solid #fff;z-index:1}details{margin-top:18px}summary{cursor:pointer;color:#bcd8f4}table{border-collapse:collapse;width:100%;font-size:13px;margin:15px 0}td,th{padding:10px;border-bottom:1px solid #344152;text-align:left}th{position:sticky;top:0;background:#18202c}.tablewrap{overflow:auto;max-height:420px}td button{padding:3px 6px;background:none;border:0;color:#97cfff}footer{font-size:13px;margin:25px 0;color:#aab7c9}.note{font-size:13px;color:#b8c5d9;max-width:1100px}.raw .mask{display:none}.hide{display:none!important}
#loadstate{height:1.5em;line-height:1.5;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#scenes{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}#scenes.single{grid-template-columns:1fr}
@media(max-width:1000px){#scenes{grid-template-columns:1fr}}
@media(max-width:700px){body{padding:12px}.panels{gap:0}.panel{padding:5px}.panel h3{font-size:11px;min-height:30px}.timeline{grid-template-columns:auto 1fr}#stamp{grid-column:1/-1}.controls{gap:10px}h1{font-size:22px}.stats{font-size:12px}.scenehead{padding:10px}}
</style>
<header><h1>Kitchen 1711 · SAM and Blender</h1><div class="muted">100 shared video timestamps · original authored cameras · concept-level mask review</div>
<p class="warning"><strong>Diagnostic masks, pending identity review.</strong> Each Blender mask joins all entities with the selected concept. SAM may detect more real items, miss an item, or switch tracks. These are not individual-object scores or verified ground truth. <span id="quality"></span></p></header>
<div class="controls"><label>Concept <select id="concept" aria-label="Concept"></select></label><label>Scenes <select id="scene"><option value="both">Both scenes</option><option value="before_vqa">Original before VQA</option><option value="vggt_guided">VGGT guided</option></select></label><label>Opacity <input id="opacity" aria-label="Mask opacity" type="range" min="0" max="1" step=".05" value=".56"></label><label><input type="checkbox" id="raw"> RGB only</label></div>
<div class="timeline"><button id="play">▶ Play</button><input id="position" type="range" min="0" value="0" aria-label="Selected video frame"><div id="stamp"></div></div>
<div id="loadstate" class="note" role="status" aria-live="polite"></div><div id="trend" aria-label="Per-frame IoU timeline"></div><div class="note">Timeline bars: <span style="color:#e9a567">original</span> / <span style="color:#78b6f1">guided</span> IoU. Click a bar or use ← → to step. Playback follows source timestamps.</div>
<div class="legend"><span><i class="dot sam"></i>SAM only</span><span><i class="dot blender"></i>Blender only</span><span><i class="dot overlap"></i>Overlap</span></div><div id="reference" class="ref"></div><div id="reviewnotes" class="warning hide"></div>
<main id="scenes"></main>
<details><summary>All concept summaries and metric definitions</summary><p class="note">Mean IoU averages frames with a nonempty union; both-empty frames are excluded. Pixel-pooled IoU weights larger masks more. Precision = intersection / Blender area; recall = intersection / SAM area. All values use full-resolution masks. Scene-wide aggregates are inventory-dependent, so comparing their totals is not a controlled reconstruction ranking.</p><div id="totals" class="note"></div><div class="tablewrap"><table><thead><tr><th>Concept</th><th>Original mean IoU</th><th>Guided mean IoU</th><th>Shared SAM reference</th></tr></thead><tbody id="summary"></tbody></table></div></details>
<footer><a href="per_frame.csv">Per-frame CSV</a> · <a href="per_concept.csv">Per-concept CSV</a> · <a href="review.json">Full report and provenance</a><p>Next: review tracks and associate physical entities before individual-object scoring. Small preview masks use nearest-neighbor resizing; scores use the original pixel grid.</p></footer>
<script id="data" type="application/json">__REPORT__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),$=id=>document.getElementById(id),F=D.frames;
const names=[...new Set(D.scenes.flatMap(s=>Object.keys(s.concepts)))].sort(),fmt=x=>x==null?'—':(100*x).toFixed(1)+'%',num=x=>x.toLocaleString();
let pos=0,playing=false,startClock=0,startTime=0,raf=null,drawVersion=0;
const assetCache=new Map();
function ready(url){if(assetCache.has(url))return assetCache.get(url);const promise=new Promise((resolve,reject)=>{const image=new Image();image.onload=()=>{if(image.decode)image.decode().then(resolve,reject);else resolve()};image.onerror=()=>reject(new Error('Missing or unreadable preview: '+url));image.src=url});assetCache.set(url,promise);if(assetCache.size>96)assetCache.delete(assetCache.keys().next().value);return promise}
for(const name of names){const option=document.createElement('option');option.value=name;option.textContent=name.replaceAll('_',' ');$('concept').append(option)}
$('concept').value=names.includes('microwave')?'microwave':names[0];$('position').max=F.length-1;
pos=Math.max(0,F.findIndex((_,i)=>D.scenes.every(s=>{const row=s.concepts[$('concept').value]?.frames[i];return row&&row.sam_area>0&&row.blender_area>0})));
for(const scene of D.scenes){const el=document.createElement('section');el.className='scene';el.id='s-'+scene.key;el.innerHTML='<div class="scenehead"><h2></h2><div class="stats"></div></div><div class="missing hide">This scene has no entities for this concept.</div><div class="panels"><div class="panel"><h3>Original video + SAM concept mask</h3><div class="image"><img alt="Source video frame"><div class="mask samLayer"></div></div></div><div class="panel"><h3>Blender render + entity union</h3><div class="image"><img alt="Authored Blender render"><div class="mask blenderLayer"></div></div></div><div class="panel"><h3>Mask overlap on original video</h3><div class="image"><img alt="Source with mask overlap"><div class="mask diffLayer"></div></div></div></div><div class="details"></div>';
el.querySelector('h2').textContent=scene.label;el.querySelectorAll('.image').forEach(i=>{i.style.maxWidth=(475*D.width/D.height)+'px';i.style.aspectRatio=D.width+'/'+D.height});$('scenes').append(el)}
function stop(){playing=false;cancelAnimationFrame(raf);$('play').textContent='▶ Play'}
function setPosition(index){pos=Math.max(0,Math.min(F.length-1,index));if(playing){startClock=performance.now();startTime=F[pos].time_seconds}draw()}
async function draw(){const version=++drawVersion,displayPos=pos,name=$('concept').value,chosenScene=$('scene').value,frame=F[displayPos];
const needed=[frame.rgb];for(const scene of D.scenes){const c=scene.concepts[name];if(c&&(chosenScene==='both'||chosenScene===scene.key))needed.push('rgb/'+scene.key+'/'+String(frame.index).padStart(6,'0')+'.jpg',c.frames[displayPos].sprite)}
$('loadstate').textContent='Loading synchronized frame…';try{await Promise.all(needed.map(ready))}catch(error){if(version===drawVersion){stop();$('loadstate').textContent=error.message}return}if(version!==drawVersion)return;$('loadstate').textContent='';
$('position').value=displayPos;$('stamp').textContent=frame.time_seconds.toFixed(3)+' s · source '+frame.index+' · '+(displayPos+1)+'/'+F.length;
$('scenes').classList.toggle('single',chosenScene!=='both');
const shared=D.shared_references[name];$('reference').textContent=shared?(shared.reference_comparable?'Shared SAM reference: identical decoded masks, prompt and anchor in both runs.':'Shared SAM reference differs: '+shared.differing_frame_count+' frames differ'+(shared.same_text_and_anchor?'':'; prompt text or anchor differs')+'. Compare cautiously.'):'Scene-specific concept: not present in both inventories.';
const notes=D.review_notes.notes.filter(n=>n.concept===name);$('reviewnotes').replaceChildren();$('reviewnotes').classList.toggle('hide',!notes.length);for(const note of notes){const p=document.createElement('p');p.textContent='Sampled review: '+note.finding;$('reviewnotes').append(p)}if(notes.length&&D.review_notes.review_url){const a=document.createElement('a');a.href=D.review_notes.review_url;a.textContent='See sampled track review; full temporal mapping remains pending';$('reviewnotes').append(a)}
for(const scene of D.scenes){const el=$('s-'+scene.key),concept=scene.concepts[name];el.classList.toggle('hide',chosenScene!=='both'&&chosenScene!==scene.key);el.querySelector('.panels').classList.toggle('hide',!concept);el.querySelector('.missing').classList.toggle('hide',!!concept);
if(!concept){el.querySelector('.stats').textContent='';el.querySelector('.details').textContent='';continue}const row=concept.frames[displayPos];
const image=el.querySelectorAll('img');image[0].src=frame.rgb;image[1].src='rgb/'+scene.key+'/'+String(frame.index).padStart(6,'0')+'.jpg';image[2].src=frame.rgb;
el.querySelectorAll('.mask').forEach(m=>m.style.backgroundImage='url("'+row.sprite+'")');
el.querySelector('.stats').textContent='IoU '+fmt(row.iou)+' · Dice '+fmt(row.dice)+' · precision '+fmt(row.precision_vs_sam)+' · recall '+fmt(row.recall_vs_sam);
el.querySelector('.details').replaceChildren();const details=el.querySelector('.details');const text=document.createElement('span');text.textContent='SAM '+num(row.sam_area)+' px / Blender '+num(row.blender_area)+' px · '+row.sam_instances+' SAM instances · '+row.presence.replaceAll('_',' ')+' · '+concept.entities.length+' Blender entities: '+concept.entities.join(', ')+'. ';details.append(text);const link=document.createElement('a');link.href=scene.catalog;link.textContent='Review SAM tracks';details.append(link)}
$('trend').querySelectorAll('.bar').forEach((bar,i)=>bar.classList.toggle('current',i===displayPos))}
function timeline(){const name=$('concept').value;$('trend').replaceChildren();F.forEach((frame,i)=>{const bar=document.createElement('button');bar.className='bar';bar.title=frame.time_seconds.toFixed(3)+' s';bar.setAttribute('aria-label','Frame '+frame.index);bar.onclick=()=>setPosition(i);for(const s of D.scenes){const v=s.concepts[name]?.frames[i].iou;const line=document.createElement('span');line.style.height=v==null?'0':Math.max(2,v*100)+'%';line.style.opacity=v==null?'0':'1';bar.append(line)}$('trend').append(bar)});draw()}
function tick(now){if(!playing)return;const t=startTime+(now-startClock)/1000;let next=pos;while(next+1<F.length&&F[next+1].time_seconds<=t)next++;if(next!==pos){pos=next;draw()}if(t>=F[F.length-1].time_seconds){stop();return}raf=requestAnimationFrame(tick)}
$('play').onclick=()=>{if(playing){stop();return}if(pos===F.length-1)pos=0;playing=true;startClock=performance.now();startTime=F[pos].time_seconds;$('play').textContent='Ⅱ Pause';draw();raf=requestAnimationFrame(tick)};
$('position').oninput=e=>setPosition(Number(e.target.value));$('concept').onchange=timeline;$('scene').onchange=draw;$('opacity').oninput=e=>document.documentElement.style.setProperty('--opacity',e.target.value);$('raw').onchange=e=>document.body.classList.toggle('raw',e.target.checked);
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON'].includes(document.activeElement.tagName))return;if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();stop();setPosition(pos+(e.key==='ArrowLeft'?-1:1))}});
for(const name of names){const tr=document.createElement('tr'),cell=document.createElement('td'),button=document.createElement('button');button.textContent=name;button.onclick=()=>{$('concept').value=name;timeline();window.scrollTo({top:0,behavior:'smooth'})};cell.append(button);tr.append(cell);for(const scene of D.scenes){const td=document.createElement('td');td.textContent=fmt(scene.concepts[name]?.summary.mean_frame_iou);tr.append(td)}const td=document.createElement('td'),r=D.shared_references[name];td.textContent=!r?'Scene-specific':r.reference_comparable?'Identical':'Differs';tr.append(td);$('summary').append(tr)}
$('totals').textContent=D.scenes.map(s=>s.label+': '+s.concept_count+' concepts / '+s.entity_count+' entities; all-concept mean IoU '+fmt(s.summary.mean_frame_iou)+'; pooled IoU '+fmt(s.summary.pixel_pooled_iou)).join(' | ');
$('quality').textContent=D.scenes.map(s=>s.label+': '+s.zero_detection_concepts.length+' concepts have no SAM detections in any selected frame.').join(' ')+' Empty SAM masks do not establish that an object is absent.';
timeline();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."), help="Project root with paired 1711 artifacts")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview-height", type=int, default=480)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
