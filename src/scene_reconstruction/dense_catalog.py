"""Conservative whole-track deduplication and an offline, unreviewed SAM viewer."""
from __future__ import annotations

import colorsys
from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .common import load_manifest, write_json

_BITS = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
_NOTE = ("Unreviewed SAM predictions; no Blender score. Empty predictions do not establish "
         "object absence. Deduplication is a coarse-grid heuristic; parts and whole objects "
         "remain separate unless their masks agree over time. Raw tracks are preserved.")


def _pair_metrics(first, second, presence_a, presence_b, threshold):
    """Compare packed masks; visibility comes from full-resolution nonempty masks."""
    common = presence_a & presence_b
    either = presence_a | presence_b
    intersections = _BITS[np.bitwise_and(first[common], second[common])].sum(axis=1)
    unions = _BITS[np.bitwise_or(first[common], second[common])].sum(axis=1)
    # A small object disappearing during resize supplies no overlap evidence.
    valid = unions > 0
    ious = intersections[valid] / unions[valid]
    return {"common_visible_frames": int(common.sum()), "measurable_common_frames": len(ious),
            "presence_overlap": float(common.sum() / either.sum()) if either.any() else 0.0,
            "median_iou": float(np.median(ious)) if len(ious) else None,
            "fraction_above_threshold": float(np.mean(ious >= threshold)) if len(ious) else 0.0}


def _groups(ids, pairs):
    """Deterministic complete-link grouping; never select per-frame winners."""
    groups, conflicts = [], []
    for identity in ids:
        eligible, partial = [], []
        for group in groups:
            agrees = [pairs[tuple(sorted((identity, other)))]["duplicate"] for other in group]
            if all(agrees):
                eligible.append(group)
            elif any(agrees):
                partial.append(group)
        if len(eligible) == 1 and not partial:
            eligible[0].append(identity)
        else:
            if eligible or partial:
                conflicts.append({"track": identity, "candidate_groups": [g[0] for g in eligible + partial],
                                  "reason": "ambiguous_or_incomplete_link"})
            groups.append([identity])
    return groups, conflicts


def _html(catalog):
    embedded = json.dumps(catalog, allow_nan=False).replace("<", "\\u003c")
    return """<!doctype html><meta charset="utf-8"><title>Dense SAM track review</title>
<style>body{font:15px system-ui;background:#111824;color:#e9eef6;margin:20px;max-width:1100px}
select,input,button{font:inherit;margin:6px}a{color:#8cd8ff}.views{display:flex;gap:20px;flex-wrap:wrap}
.picture{position:relative;width:270px;max-width:40vw}.picture img{display:block;width:100%}
.picture .mask{position:absolute;inset:0}#loading{height:1.5em;line-height:1.5;white-space:nowrap}
#details{min-height:3em}.snippets{display:flex;gap:10px;flex-wrap:wrap}.snippet{max-width:130px}
.snippet .picture{width:120px}input[type=range]{width:70%}label{display:inline-block}</style>
<h1>Dense SAM tracks</h1><p id="note"></p><p id="summary"></p>
<label>Canonical track <select id="track"></select></label>
<label>Raw track <select id="alias"></select></label><button id="play">Play</button>
<div><input id="frame" type="range" min="0" value="0"><span id="time"></span></div>
<div id="loading" role="status"></div><div class="views"><div><p>Source</p>
<div class="picture"><img id="source" alt="Source frame"></div></div>
<div><p>Source + predicted mask</p><div class="picture"><img id="base" alt="Source frame">
<img id="mask" class="mask" alt="Track mask"></div></div></div>
<p id="details"></p><h2>Raw aliases and discovery seeds</h2><div id="snippets" class="snippets"></div>
<p><a href="catalog.json">Catalog data</a> · <a href="../raw_tracks.json">All raw tracks</a></p>
<script>const D=""" + embedded + """;
const el=id=>document.getElementById(id), ts=el('track'), as=el('alias'), fs=el('frame');
el('note').textContent=D.interpretation;
el('summary').textContent=`${D.raw_track_count} raw tracks → ${D.tracks.length} canonical tracks; ${D.ambiguous_pairs.length} uncertain pairs, ${D.grouping_conflicts.length} grouping conflicts. ${D.discovery.reduce((n,a)=>n+a.budget_deferred,0)} proposals deferred by tracking budget. Matching grid: ${D.matching.width}×${D.matching.height}.`;
fs.max=Math.max(0,D.frames.length-1);let generation=0,timer=null;
const option=(value,label)=>{const o=document.createElement('option');o.value=value;o.textContent=label;return o};
D.tracks.forEach(t=>ts.append(option(t.id,`${t.id} (${t.aliases.length} raw)`)));
function show(){const f=D.frames[+fs.value],t=D.tracks.find(t=>t.id===ts.value),a=t?.aliases.find(a=>a.id===as.value);
 if(!f)return; const token=++generation;el('loading').textContent='Loading synchronized frame…';
 el('time').textContent=`Frame ${f.index} · ${f.time_seconds.toFixed(2)} s`;
 const pairs=[['source',f.preview],['base',f.preview]];if(a)pairs.push(['mask',a.overlays[+fs.value]]);
 el('mask').style.display=a?'block':'none';let pending=pairs.length;
 pairs.forEach(([id,url])=>{const im=el(id);im.onload=()=>{if(token===generation&&!--pending)el('loading').textContent=''};
 im.onerror=()=>{if(token===generation)el('loading').textContent='Preview could not be loaded'};im.src=url});
 el('details').textContent=a?`${a.id}; seed frame ${a.anchor_frame}; ${a.areas[+fs.value]} predicted pixels${a.areas[+fs.value]?'':' — empty prediction, absence unverified'}. Canonical representative: ${t.representative}.`: 'No tracks discovered.';
}
function choose(){const t=D.tracks.find(t=>t.id===ts.value);as.replaceChildren();el('snippets').replaceChildren();
 if(t)t.aliases.forEach(a=>{as.append(option(a.id,a.id===t.representative?`${a.id} (representative)`:a.id));
 const box=document.createElement('div');box.className='snippet';const link=document.createElement('a');link.href='../'+a.index_file;link.textContent=a.id;box.append(link);
 const i=Math.max(0,D.frames.findIndex(f=>f.index===a.anchor_frame)),p=document.createElement('div');p.className='picture';
 [D.frames[i].preview,a.seed_overlay||a.overlays[i]].forEach((url,j)=>{const im=document.createElement('img');im.src=url;if(j)im.className='mask';p.append(im)});box.append(p);
 const seed=document.createElement('a');seed.href='../'+(a.seed_mask||a.proposal_file||'raw_tracks.json');seed.textContent=`Seed ${a.proposal_id}`;box.append(seed);el('snippets').append(box)});show()}
ts.onchange=choose;as.onchange=show;fs.oninput=show;el('play').onclick=()=>{if(timer){clearInterval(timer);timer=null;el('play').textContent='Play'}else{timer=setInterval(()=>{fs.value=(+fs.value+1)%D.frames.length;show()},240);el('play').textContent='Pause'}};choose();
</script>"""


def build_catalog(output, duplicate_iou=.9, min_common_frames=3, min_presence_overlap=.8):
    """Write catalog/catalog.json and a portable HTML review; return the catalog.

    Matching uses nearest-neighbor masks bounded by 270×480, packed on disk.
    A duplicate needs >=80% of measurable common-visible frames above the IoU
    threshold in addition to median IoU, presence overlap, and frame support.
    Each canonical track is one entire raw representative, never a stitched mask.
    """
    if not 0 < duplicate_iou <= 1 or not 0 < min_presence_overlap <= 1 or min_common_frames < 1:
        raise ValueError("Invalid duplicate matching thresholds")
    output = Path(output).resolve()
    manifest = load_manifest(output / "manifest.json", allow_sparse=True)
    raw = json.loads((output / "raw_tracks.json").read_text())
    if raw.get("schema_version") != 1 or raw.get("status") != "complete":
        raise ValueError("Raw tracking must be complete before catalog construction")
    width, height = int(raw["width"]), int(raw["height"])
    if width < 1 or height < 1 or (width, height) != (manifest["width"], manifest["height"]):
        raise ValueError("Track and manifest dimensions disagree")
    scale = min(1, 270 / width, 480 / height)
    size = max(1, round(width * scale)), max(1, round(height * scale))
    destination = output / "catalog"
    (destination / "packed").mkdir(parents=True, exist_ok=True)
    (destination / "source").mkdir(exist_ok=True)
    frames = manifest["frames"]
    frame_ids = [frame["index"] for frame in frames]
    catalog_frames = []
    for frame in frames:
        preview = f"source/{frame['index']:06d}.jpg"
        with Image.open(output / frame["file"]) as image:
            if image.size != (width, height):
                raise ValueError("Source frame dimensions disagree with manifest")
            image.convert("RGB").resize(size, Image.Resampling.LANCZOS).save(destination / preview, quality=85)
        catalog_frames.append({"index": frame["index"], "time_seconds": frame["time_seconds"], "preview": preview})
    records, packed, presence = {}, {}, {}
    for track in sorted(raw["tracks"], key=lambda item: item["id"]):
        identity = track["id"]
        if identity in records or Path(identity).name != identity or identity in (".", ".."):
            raise ValueError("Track IDs must be unique simple names")
        index = json.loads((output / track["index_file"]).read_text())
        if index.get("status") != "complete" or index.get("id") != identity:
            raise ValueError(f"Incomplete or mismatched track {identity}")
        if track["anchor_frame"] not in frame_ids:
            raise ValueError(f"Track {identity} anchor is not in the source manifest")
        rows = index["frames"]
        if [row["index"] for row in rows] != frame_ids:
            raise ValueError(f"Track {identity} must contain every requested frame exactly once")
        masks, areas = [], []
        for row, frame in zip(rows, frames):
            if not np.isfinite(row["time_seconds"]) or abs(row["time_seconds"] - frame["time_seconds"]) > 1e-6:
                raise ValueError(f"Track {identity} timestamps disagree")
            with Image.open(output / row["mask"]) as image:
                values = np.asarray(image)
                if image.size != (width, height) or values.ndim != 2 or not np.isin(values, [0, 1, 255]).all():
                    raise ValueError(f"Invalid binary mask for {identity}")
                area = int(np.count_nonzero(values))
                if area != row["area_pixels"]:
                    raise ValueError(f"Mask area disagrees for {identity}")
                masks.append(np.packbits(np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 0))
                areas.append(area)
        packed_file = destination / "packed" / f"{identity}.npy"
        np.save(packed_file, np.stack(masks))
        packed[identity] = packed_file
        presence[identity] = np.array(areas) > 0
        records[identity] = dict(track, areas=areas)
        for key in ("seed_mask", "seed_quality", "seed_stability"):
            if key in index:
                records[identity][key] = index[key]
        discovery = f"discovery/{track['anchor_frame']:06d}/index.json"
        if (output / discovery).is_file():
            records[identity]["proposal_file"] = discovery
    pairs, ambiguous, duplicates = {}, [], []
    identities = list(records)
    @lru_cache(maxsize=8)
    def read_packed(identity):
        return np.load(packed[identity])
    for i, first in enumerate(identities):
        for second in identities[i + 1:]:
            stats = _pair_metrics(read_packed(first), read_packed(second), presence[first], presence[second], duplicate_iou)
            stats["duplicate"] = bool(stats["measurable_common_frames"] >= min_common_frames
                and stats["presence_overlap"] >= min_presence_overlap
                and stats["median_iou"] is not None and stats["median_iou"] >= duplicate_iou
                and stats["fraction_above_threshold"] >= .8)
            pairs[first, second] = stats
            row = dict(first=first, second=second, **stats)
            if stats["duplicate"]:
                duplicates.append(row)
            elif stats["median_iou"] is not None and stats["median_iou"] >= max(.5, duplicate_iou - .2):
                ambiguous.append(row)
    groups, conflicts = _groups(identities, pairs)
    canonical = []
    for number, group in enumerate(groups, 1):
        color = tuple(round(c * 255) for c in colorsys.hsv_to_rgb((number * .618034) % 1, .75, 1))
        aliases = []
        for identity in group:
            (destination / "overlays" / identity).mkdir(parents=True, exist_ok=True)
            overlays = []
            for ordinal, frame in enumerate(frames):
                mask = np.unpackbits(read_packed(identity)[ordinal], count=size[0] * size[1]).reshape(size[1], size[0]).astype(bool)
                rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
                rgba[mask] = (*color, 145)
                relative = f"overlays/{identity}/{frame['index']:06d}.png"
                Image.fromarray(rgba).save(destination / relative)
                overlays.append(relative)
            alias = dict(records[identity], overlays=overlays)
            if alias.get("seed_mask"):
                with Image.open(output / alias["seed_mask"]) as image:
                    mask = np.asarray(image.resize(size, Image.Resampling.NEAREST)) > 0
                rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
                rgba[mask] = (*color, 145)
                alias["seed_overlay"] = f"overlays/{identity}/seed.png"
                Image.fromarray(rgba).save(destination / alias["seed_overlay"])
            aliases.append(alias)
        canonical.append({"id": f"canonical_{number:04d}", "representative": group[0],
                          "index_file": records[group[0]]["index_file"], "color": list(color), "aliases": aliases})
    discovery = []
    for path in sorted((output / "discovery").glob("*/index.json")):
        info = json.loads(path.read_text())
        if info.get("status") == "complete":
            discovery.append({"anchor_frame": info["anchor_frame"], "index_file": str(path.relative_to(output)),
                              "hypotheses": info.get("proposal_hypotheses"), "retained": len(info.get("proposals", [])),
                              "tracking_seeds": len(info.get("tracking_proposal_ids", [])),
                              "budget_deferred": len(info.get("budget_deferred_ids", []))})
    catalog = {"schema_version": 1, "status": "complete", "interpretation": _NOTE, "discovery": discovery,
               "raw_track_count": len(identities), "frames": catalog_frames, "tracks": canonical,
               "matching": {"width": size[0], "height": size[1], "resize": "nearest",
                   "duplicate_iou": duplicate_iou, "min_common_frames": min_common_frames,
                   "min_presence_overlap": min_presence_overlap, "min_fraction_above_threshold": .8,
                   "grouping": "deterministic complete-link; one whole representative; no per-frame selection"},
               "duplicate_pairs": duplicates, "ambiguous_pairs": ambiguous, "grouping_conflicts": conflicts}
    write_json(destination / "catalog.json", catalog)
    (destination / "index.html").write_text(_html(catalog))
    return catalog
