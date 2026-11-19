"""Offline inspection report for a surface-derived global similarity.

The pictures contain sampled *actual* input points and target triangles. Their
sampling affects display only; fitting diagnostics are supplied by the caller.
No camera positions enter this module's geometry or transformation calculation.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .alignment import transform_points, validate_transform


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sample_by_label(labels, limit):
    """Keep rare anchor groups visible while capping displayed point count."""
    groups = [np.flatnonzero(labels == label) for label in np.unique(labels)]
    allocations = [0] * len(groups)
    remaining = min(limit, len(labels))
    # Water-fill small groups, redistributing their unused share to others.
    while remaining:
        available = [i for i, group in enumerate(groups) if allocations[i] < len(group)]
        if not available:
            break
        share = max(1, remaining // len(available))
        for i in available:
            amount = min(share, len(groups[i]) - allocations[i], remaining)
            allocations[i] += amount
            remaining -= amount
            if not remaining:
                break
    selected = [group[np.linspace(0, len(group) - 1, amount, dtype=int)]
                for group, amount in zip(groups, allocations) if amount]
    return np.sort(np.concatenate(selected)) if selected else np.empty(0, dtype=int)


def build_payload(report, source_points, source_labels, targets, *, max_points=12000,
                  max_triangles=3000):
    """Validate, transform and deterministically sample review geometry."""
    if not isinstance(max_points, int) or max_points < 1:
        raise ValueError("max_points must be a positive integer")
    if not isinstance(max_triangles, int) or max_triangles < 1:
        raise ValueError("max_triangles must be a positive integer")
    points = np.asarray(source_points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Report source points must be finite Nx3 coordinates")
    anchors = report.get("anchors", [])
    labels = np.full(len(points), -1, dtype=int) if source_labels is None else np.asarray(source_labels)
    if (labels.shape != (len(points),) or not np.issubdtype(labels.dtype, np.integer)
            or np.any(labels < -1) or np.any(labels >= len(anchors))):
        raise ValueError("Report labels must be integer anchor indices, or -1, for every point")
    transform = validate_transform(report["transform"])
    selection = _sample_by_label(labels, max_points)
    aligned = transform_points(points[selection], transform)
    planes = []
    for plane in targets.get("planes", []):
        triangles = np.asarray(plane.get("triangles", []), dtype=np.float64)
        if not triangles.size:
            triangles = np.empty((0, 3, 3), dtype=np.float64)
        if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.isfinite(triangles).all():
            raise ValueError("Report target triangles must be finite Mx3x3 coordinates")
        count = len(triangles)
        if count > max_triangles:
            triangles = triangles[np.linspace(0, count - 1, max_triangles, dtype=int)]
        planes.append({"name": str(plane.get("name", plane.get("key", "Surface"))),
                       "display_name": str(plane.get("display_name") or plane.get("name", plane.get("key", "Surface"))),
                       "triangles": triangles.round(6).tolist(), "total_triangles": count})
    summary = {key: report.get(key) for key in
               ("status", "anchors", "diagnostics", "camera_metrics", "warnings")}
    summary["transform"] = transform
    summary["target_profile"] = report.get("target_profile") or targets.get("profile")
    summary["candidates"] = report.get("candidates", [])[:100]
    summary["candidate_count"] = len(report.get("candidates", []))
    return _jsonable({"report": summary, "points": aligned.round(6).tolist(),
                      "labels": labels[selection].tolist(), "source_indices": selection.tolist(),
                      "source_point_count": len(points), "planes": planes,
                      "display_note": "Display sampling only; numeric fit diagnostics are unchanged."})


def write_report(path, report, source_points, source_labels, targets, *, max_points=12000,
                 max_triangles=3000):
    """Write an atomic standalone HTML report; return its resolved path.

    ``source_points`` are raw VGGT coordinates. ``source_labels`` identifies
    report anchor indexes (or -1 for unassigned points). Target triangles are
    already in Blender-world coordinates. ``max_triangles`` is per surface.
    """
    payload = build_payload(report, source_points, source_labels, targets,
                            max_points=max_points, max_triangles=max_triangles)
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    # Data lives in a script text node; escaping '<' prevents </script> injection.
    encoded = (encoded.replace("&", "\\u0026").replace("<", "\\u003c")
               .replace(">", "\\u003e").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029"))
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_HTML.replace("__SURFACE_DATA__", encoded), encoding="utf-8")
    temporary.replace(path)
    return path


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Surface alignment review</title>
<style>
:root{color-scheme:light dark;--bg:#f9fafb;--fg:#18212c;--muted:#596674;--line:#cbd3dc;--panel:#fff;--warn:#794100;--warn-bg:#fff0d8;--s0:#087e8b;--s1:#b34b17;--s2:#7554b7;--s3:#39752c;--s4:#b43b76;--s5:#2766ba;--context:#8b939c}
@media(prefers-color-scheme:dark){:root{--bg:#111720;--fg:#e2e8ee;--muted:#a9b6c4;--line:#3c4c5c;--panel:#1b2531;--warn:#ffd59b;--warn-bg:#392c1c;--s0:#50c3cb;--s1:#f7aa6d;--s2:#b9a3eb;--s3:#a0ce85;--s4:#f49bc2;--s5:#8fbafb;--context:#677a8c}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif}main{max-width:1200px;margin:auto;padding:24px}h1{font-size:1.6rem;margin:0 0 8px}h2{font-size:1.1rem;margin:24px 0 8px}p{margin:8px 0}a{color:var(--s5)}.muted{color:var(--muted)}.notice{padding:12px 16px;background:var(--warn-bg);color:var(--warn);border-left:3px solid var(--warn)}.controls,.legend,.links{display:flex;flex-wrap:wrap;gap:12px;align-items:center}.controls{margin:16px 0}.controls label{display:flex;align-items:center;gap:6px}select,button{font:inherit;background:var(--panel);color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:6px 9px}input{accent-color:var(--s0)}svg{display:block;width:100%;height:520px;background:var(--panel);border:1px solid var(--line)}svg text{fill:var(--fg);font:13px system-ui,sans-serif}.legend{margin:10px 0}.legend span{display:inline-flex;gap:6px;align-items:center}.dot{width:9px;height:9px;border-radius:50%;display:inline-block}.wire{display:inline-block;width:20px;border-top:2px solid var(--fg)}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;border-bottom:1px solid var(--line);padding:8px;white-space:nowrap}th{font-weight:600}.metrics{display:flex;flex-wrap:wrap;gap:24px}.metric strong{display:block;font-size:1.2rem}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:var(--panel);padding:12px;font-size:13px}details{margin-top:16px}summary{cursor:pointer}.small{font-size:13px}.sr-only{position:absolute;width:1px;height:1px;padding:0;margin:-1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap;border:0}@media(max-width:600px){main{padding:14px}svg{height:380px}.controls{gap:10px}h1{font-size:1.35rem}}
</style></head><body><main>
<h1>Surface alignment review</h1>
<p id="status" class="notice"></p>
<p>One global rotation, translation and positive scale fitted from structural surfaces. Camera correspondences are evaluated after fitting.</p>
<p id="target-profile" class="small muted" hidden></p>
<div class="links"><a href="viewer.html">Interactive 3D viewer</a><a href="alignment.json">Alignment and camera metrics</a><a href="surface_fit.json">Surface-fit diagnostics</a></div>
<h2>Actual geometry in Blender coordinates</h2>
<div class="controls">
<label for="projection">Projection <select id="projection"><option value="xy">Top · X/Y</option><option value="xz" selected>Front · X/Z</option><option value="yz">Side · Y/Z</option></select></label>
<label><input id="show-context" type="checkbox" checked> Unassigned cloud points</label>
<label><input id="show-target" type="checkbox" checked> Blender anchor faces</label>
</div>
<svg id="geometry" role="img" aria-labelledby="geometry-title geometry-description"><title id="geometry-title">Aligned point cloud and Blender structural surfaces</title><desc id="geometry-description">Points show the surface-transformed VGGT cloud; outlines show actual Blender anchor triangles. The projection selector changes the two displayed world axes. Units are Blender units, not independently measured meters.</desc><g id="plot"></g></svg>
<div id="legend" class="legend small"></div>
<p id="sample-note" class="small muted"></p>
<h2>Anchor residuals</h2><p class="small muted">These surfaces determined the alignment. Their residuals describe fit agreement, not independent reconstruction accuracy.</p>
<div class="table-wrap"><table><thead><tr><th>Surface</th><th>Source plane</th><th>Retained points</th><th>RMS</th><th>Median</th><th>95th percentile</th><th>Plane overlap estimate</th></tr></thead><tbody id="anchor-rows"></tbody></table></div>
<p class="small muted">Point counts and distance residuals use retained points inside the finite Blender face footprint, with the fitting tolerance. Distances are in Blender units.</p>
<p class="small muted">Plane overlap estimates the fraction of the pooled, nearby parallel source detections inside that footprint, using the candidate-scoring sample. It uses a different cohort from the retained-point count. A low percentage can reflect coplanar extensions beyond the named target surface; it is not the fraction of Blender geometry reconstructed.</p>
<details><summary>Anchor support and source-plane fragments</summary>
<div class="table-wrap"><table><thead><tr><th>Surface</th><th>Source frames</th><th>Tangent width 1</th><th>Tangent width 2</th><th>Original plane fragments</th></tr></thead><tbody id="support-rows"></tbody></table></div>
<p class="small muted">Source frames are unique frame IDs among retained, spatially balanced anchor points. Tangent widths are 5th–95th percentile extents along the patch's two principal in-plane directions, in Blender units. These diagnose support; they are not independently measured object dimensions.</p>
</details>
<h2>Camera evaluation</h2><div id="camera-metrics" class="metrics"></div>
<p class="small muted">These camera positions and orientations did not determine the surface transform. VGGT geometry and its cameras still come from the same prediction.</p>
<h2>Checks and limitations</h2><ul id="warnings"></ul>
<details><summary>Fit diagnostics and competing candidates</summary><pre id="diagnostics"></pre></details>
<details><summary>Global transformation</summary><pre id="transform"></pre></details>
</main><script id="surface-data" type="application/json">__SURFACE_DATA__</script>
<script>
(() => {
  'use strict';
  const data=JSON.parse(document.getElementById('surface-data').textContent), report=data.report;
  const get=id=>document.getElementById(id), svg=get('geometry'), plot=get('plot');
  const ns='http://www.w3.org/2000/svg', anchors=report.anchors||[], colors=['--s0','--s1','--s2','--s3','--s4','--s5'];
  const color=i=>`var(${i<0?'--context':colors[i%colors.length]})`;
  const finite=value=>typeof value==='number'&&Number.isFinite(value);
  const number=(value,digits=4)=>finite(value)?value.toLocaleString(undefined,{maximumFractionDigits:digits}):'—';
  const anchorName=(anchor,i)=>anchor?.display_name||anchor?.name||`Surface ${i+1}`;
  const text=(tag,value,parent,className)=>{const el=document.createElement(tag);el.textContent=value;if(className)el.className=className;parent.appendChild(el);return el;};
  const line=(tag,attrs,parent=plot)=>{const el=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([key,value])=>el.setAttribute(key,String(value)));parent.appendChild(el);return el;};
  const accepted=['accepted','accepted_automatic'].includes(report.status);
  if(report.status==='automatic_candidate')get('status').textContent='Passed configured geometric checks; semantic match remains an automatic hypothesis. Inspect the overlay and camera evaluation.';
  else if(report.status==='needs_review')get('status').textContent='Needs review: one or more configured geometric checks did not pass. Inspect the quality issues and competing candidates before relying on this alignment.';
  else get('status').textContent=accepted?'Automatic surface match accepted by the configured geometric checks; inspect the overlay and independent evaluation.':`Automatic surface-alignment hypothesis · ${report.status||'unverified'}. Review ambiguity and diagnostics before treating the correspondence as established.`;
  if(report.target_profile){
    const profiles={named_kitchen_structural_planes_v1:'Kitchen structural surfaces',named_living_dining_structural_planes_v1:'Living/dining-room structural surfaces'};
    get('target-profile').textContent=`Target profile: ${profiles[report.target_profile]||report.target_profile}`;
    get('target-profile').hidden=false;
  }
  anchors.forEach((anchor,i)=>{
    const row=document.createElement('tr');
    const coverage=finite(anchor.coverage)?number(anchor.coverage*100,1)+'%':'—';
    [anchorName(anchor,i),anchor.source_plane_index??'—',number(anchor.point_count,0),number(anchor.rms),number(anchor.median),number(anchor.p95),coverage].forEach(value=>text('td',String(value),row));
    get('anchor-rows').appendChild(row);
    const support=document.createElement('tr'),widths=anchor.widths_90_percent_blender_units||[];
    [anchorName(anchor,i),number(anchor.frame_count,0),number(widths[0],3),number(widths[1],3),(anchor.raw_source_plane_indices||[anchor.source_plane_index]).filter(value=>value!=null).join(', ')||'—'].forEach(value=>text('td',String(value),support));
    get('support-rows').appendChild(support);
    const item=text('span','',get('legend'));const marker=text('i','',item,'dot');marker.style.background=color(i);marker.setAttribute('aria-hidden','true');text('span',anchorName(anchor,i),item);
  });
  const wire=text('span','',get('legend'));text('i','',wire,'wire').setAttribute('aria-hidden','true');text('span','Blender faces',wire);
  get('sample-note').textContent=`Displaying ${number(data.points.length,0)} of ${number(data.source_point_count,0)} supplied cloud points. ${data.display_note} Target faces are real mesh triangles; unobserved extensions are not scored as missing reconstruction.`;
  const camera=report.camera_metrics?.all||report.camera_metrics?.evaluation||report.camera_metrics||{};
  const position=camera.translation_blender_units||{}, rotation=camera.orientation_degrees||{};
  [['Camera pairs',position.count,0],['Position RMSE · Blender units',position.rmse,4],['Median orientation error · degrees',rotation.median,3]].forEach(([label,value,digits])=>{
    const box=text('div','',get('camera-metrics'),'metric');text('span',label,box,'muted');text('strong',number(value,digits),box);
  });
  const warnings=[...(report.warnings||[])];
  if(!warnings.length)warnings.push('Inspect surface identities and geometric support; passing numerical checks alone does not establish physical accuracy.');
  warnings.forEach(value=>text('li',String(value),get('warnings')));
  get('diagnostics').textContent=JSON.stringify({diagnostics:report.diagnostics,candidate_count:report.candidate_count,candidates:report.candidates},null,2);
  get('transform').textContent=JSON.stringify(report.transform,null,2);
  function draw(){
    plot.replaceChildren();
    const width=Math.max(280,svg.getBoundingClientRect().width),height=width<600?380:520;
    svg.setAttribute('viewBox',`0 0 ${width} ${height}`);svg.setAttribute('height',height);
    const projection=get('projection').value,axes={xy:[0,1],xz:[0,2],yz:[1,2]}[projection],names=['X','Y','Z'];
    const includeContext=get('show-context').checked,includeTarget=get('show-target').checked;
    const selected=data.points.map((p,i)=>({p,label:data.labels[i]})).filter(item=>includeContext||item.label>=0);
    const faces=includeTarget?data.planes.flatMap(plane=>plane.triangles):[];
    const bounds=selected.map(item=>item.p).concat(faces.flat());
    if(!bounds.length){const message=line('text',{x:20,y:40});message.textContent='No geometry selected';return;}
    let lo=[Infinity,Infinity],hi=[-Infinity,-Infinity];
    bounds.forEach(p=>axes.forEach((axis,i)=>{lo[i]=Math.min(lo[i],p[axis]);hi[i]=Math.max(hi[i],p[axis]);}));
    const pad=56,extent=hi.map((value,i)=>Math.max(value-lo[i],1e-4));
    const scale=Math.min((width-2*pad)/extent[0],(height-2*pad)/extent[1]);
    const center=lo.map((value,i)=>(value+hi[i])/2),screen=p=>[width/2+(p[axes[0]]-center[0])*scale,height/2-(p[axes[1]]-center[1])*scale];
    for(let i=0;i<5;i++){
      const fraction=i/4,x=pad+fraction*(width-2*pad),y=height-pad-fraction*(height-2*pad);
      line('line',{x1:x,y1:pad,x2:x,y2:height-pad,stroke:'var(--line)','stroke-width':.6});
      line('line',{x1:pad,y1:y,x2:width-pad,y2:y,stroke:'var(--line)','stroke-width':.6});
      let label=line('text',{x,y:height-pad+20,'text-anchor':'middle'});label.textContent=number(center[0]+(x-width/2)/scale,2);
      label=line('text',{x:pad-8,y:y+4,'text-anchor':'end'});label.textContent=number(center[1]-(y-height/2)/scale,2);
    }
    const clouds=document.createDocumentFragment();
    selected.sort((a,b)=>(a.label>=0)-(b.label>=0)).forEach(item=>{
      const [x,y]=screen(item.p);const point=line('circle',{cx:x,cy:y,r:item.label<0?1:1.4,fill:color(item.label),opacity:item.label<0?.22:.8},clouds);
      const title=line('title',{},point);title.textContent=`${item.label<0?'Unassigned':anchorName(anchors[item.label],item.label)} · (${item.p.map(v=>number(v,4)).join(', ')})`;
    });plot.appendChild(clouds);
    faces.forEach(face=>line('polygon',{points:face.map(p=>screen(p).join(',')).join(' '),fill:'none',stroke:'var(--fg)','stroke-width':.7,opacity:.45}));
    let label=line('text',{x:width/2,y:height-10,'text-anchor':'middle'});label.textContent=`${names[axes[0]]} · Blender units`;
    label=line('text',{x:pad,y:22});label.textContent=`${names[axes[1]]} · Blender units`;
  }
  ['projection','show-context','show-target'].forEach(id=>get(id).addEventListener('change',draw));
  new ResizeObserver(draw).observe(svg);draw();
})();
</script></body></html>
'''
