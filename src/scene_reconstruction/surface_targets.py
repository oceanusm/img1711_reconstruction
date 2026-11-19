"""Extract actual structural faces using explicit Blender object-name profiles.

Kitchen and living/dining profiles identify a floor, a raised horizontal
surface, and two walls. This is not general semantic recognition.
Actual triangle normals and offsets determine the selected surface; neither
camera poses nor object bounding boxes participate. Objects not matched by this
profile are never silently substituted for an absent structural anchor.
"""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np


PATTERNS = {
    "floor": (r"^oak flooring\b", r"^floor plank\b", r"^pale oak plank flooring$"),
    "counter": (r"^(left|right) of range stone counter$", r"^window return stone slab with sink opening$", r"^l-shaped countertop\b",
                r"^countertop\s*\|\s*right of range$", r"^l-shaped sink and preparation worktop$",
                r"^spice counter slab$"),
    "back": (r"^(range|stove).*backsplash\b", r"^main stone backsplash$"),
    "side": (r"^return wall backsplash\b", r"^return wall stone below window$",
             r"^sink backsplash\b", r"^sink wall backsplash$"),
}

LIVING_DINING_PATTERNS = {
    "floor": (r"^oak_plank_",),
    "counter": (r"^dining_oak_slab_top$",),
    "back": (r"^art_and_dining_wall$",),
    "side": (r"^window_wall_(?:pier_|lintel$|base$)",),
}

PROFILES = {
    "kitchen": {
        "identifier": "named_kitchen_structural_planes_v1",
        "patterns": PATTERNS,
        "labels": {"floor": "Floor", "counter": "Main countertop",
                   "back": "Range backsplash", "side": "Return backsplash"},
        "holdouts": {
            "island_top": ("Island top", (r"^island(?:\s*\|)?\s+quartz top$",)),
        },
    },
    "living_dining": {
        "identifier": "named_living_dining_structural_planes_v1",
        "patterns": LIVING_DINING_PATTERNS,
        "labels": {"floor": "Floor", "counter": "Dining tabletop",
                   "back": "Art/dining wall", "side": "Window wall"},
        "holdouts": {
            "coffee_table_top": ("Coffee tabletop", (r"^coffee_table_rounded_oak_top$",)),
            "console_top": ("Console top", (r"^console_long_oak_top$",)),
        },
    },
}


def _matches(name, patterns):
    return any(re.search(pattern, str(name), re.IGNORECASE) for pattern in patterns)


def _select_profile(profile, names, ids):
    if profile not in {"auto", *PROFILES}:
        raise ValueError(f"Unknown surface profile {profile!r}; choose auto, kitchen, or living_dining")
    if profile != "auto":
        return profile, PROFILES[profile]
    # A name without any nondegenerate triangles is not surface coverage.
    available = names[np.unique(ids)]
    missing = {key: [role for role, patterns in spec["patterns"].items()
                     if not any(_matches(name, patterns) for name in available)]
               for key, spec in PROFILES.items()}
    complete = [key for key, roles in missing.items() if not roles]
    if len(complete) > 1:
        raise ValueError("Ambiguous automatic surface profile: complete name coverage for "
                         + ", ".join(complete) + ". Select a profile explicitly.")
    if not complete:
        details = "; ".join(f"{key} missing {', '.join(roles)}" for key, roles in missing.items())
        raise ValueError("No complete named surface profile: " + details
                         + ". Inspect Blender object names; camera alignment is not a fallback.")
    key = complete[0]
    return key, PROFILES[key]


def _triangles(mesh):
    vertices = np.asarray(mesh["vertices"], dtype=np.float64)
    triangles = np.asarray(mesh["triangles"])
    ids = np.asarray(mesh["triangle_object_ids"])
    names = np.asarray(mesh["object_names"])
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError("Target mesh vertices must be finite Nx3")
    if (triangles.ndim != 2 or triangles.shape[1] != 3
            or not np.issubdtype(triangles.dtype, np.integer)
            or not triangles.size or triangles.min() < 0 or triangles.max() >= len(vertices)):
        raise ValueError("Target mesh triangles must contain valid integer vertex indices")
    if (ids.shape != (len(triangles),) or not np.issubdtype(ids.dtype, np.integer)
            or names.ndim != 1 or not len(names) or ids.min() < 0 or ids.max() >= len(names)):
        raise ValueError("Target mesh object IDs and names are inconsistent")
    xyz = vertices[triangles]
    crosses = np.cross(xyz[:, 1] - xyz[:, 0], xyz[:, 2] - xyz[:, 0])
    twice_area = np.linalg.norm(crosses, axis=1)
    valid = twice_area > np.finfo(float).eps * 100
    if not valid.any():
        raise ValueError("Target mesh has no nondegenerate triangles")
    return xyz[valid], ids[valid], names.astype(str), crosses[valid] / twice_area[valid, None], twice_area[valid] / 2


def _group(xyz, ids, names, normals, areas, patterns, label):
    object_ids = [i for i, name in enumerate(names)
                  if _matches(name, patterns)]
    chosen = np.isin(ids, object_ids)
    if not chosen.any():
        raise ValueError(
            f"Missing named {label} surface for the selected profile. "
            "Inspect Blender object names or provide a different surface profile; "
            "camera alignment is not used as a fallback."
        )
    p, n, a = xyz[chosen], normals[chosen], areas[chosen]
    eigvalues, eigvectors = np.linalg.eigh(np.einsum("n,ni,nj->ij", a, n, n))
    if eigvalues[-1] < 1.25 * max(eigvalues[-2], np.finfo(float).eps):
        raise ValueError(f"Named {label} objects do not define one dominant surface direction")
    return {"xyz": p, "ids": ids[chosen], "normals": n, "areas": a,
            "normal": eigvectors[:, -1], "centroid": np.average(p.mean(axis=1), axis=0, weights=a),
            "eigenvalues": eigvalues.tolist()}


def _outer_face(group, direction, names, label, tolerance):
    """Choose the outward-most substantial coplanar face along direction.

    Normal *signs* from triangle winding are ignored. The supplied direction is
    upward or toward the floor interior. Bevel and edge faces are rejected by
    their normal; tiny bevel slivers cannot win the plane selection by offset.
    """
    direction = np.asarray(direction, dtype=float)
    direction = direction / np.linalg.norm(direction)
    aligned = np.abs(np.einsum("ij,j->i", group["normals"], direction)) >= 1 - 1e-5
    selected = np.flatnonzero(aligned)
    if not len(selected):
        raise ValueError(f"No flat {label} faces agree with the structural normal")
    offsets = np.einsum("ij,j->i", group["xyz"][selected].mean(axis=1), direction)
    order = np.argsort(offsets, kind="stable")
    buckets = []
    for i in order:
        value = float(offsets[i])
        if not buckets or value - buckets[-1]["minimum"] > tolerance:
            buckets.append({"minimum": value, "indices": []})
        buckets[-1]["indices"].append(int(selected[i]))
    for bucket in buckets:
        ii = bucket["indices"]
        bucket["area"] = float(group["areas"][ii].sum())
        bucket["offset"] = float(np.average(np.einsum("ij,j->i", group["xyz"][ii].mean(axis=1), direction),
                                            weights=group["areas"][ii]))
    largest = max(bucket["area"] for bucket in buckets)
    substantial = [bucket for bucket in buckets if bucket["area"] >= largest * .1]
    front = max(substantial, key=lambda bucket: bucket["offset"])
    indices = front["indices"]
    p = group["xyz"][indices]
    a = group["areas"][indices]
    centroid = np.average(p.mean(axis=1), axis=0, weights=a)
    residuals = np.einsum("ij,j->i", p.reshape(-1, 3), direction) - front["offset"]
    if np.max(np.abs(residuals)) > tolerance * 2:
        raise ValueError(f"Selected {label} face is not planar at the mesh tolerance")
    object_ids = sorted(set(int(x) for x in group["ids"][indices]))
    return {"name": label, "normal": direction.tolist(), "offset": front["offset"],
            "triangles": p.tolist(), "triangle_count": len(p),
            "object_ids": object_ids, "object_names": [str(names[i]) for i in object_ids],
            "area": front["area"], "centroid": centroid.tolist(),
            "max_planarity_error": float(np.max(np.abs(residuals))),
            "parallel_face_candidates": [{"offset": b["offset"], "area": b["area"]}
                                         for b in substantial],
            "selection": "outermost substantial actual face in the oriented normal direction"}


def extract_target_planes(blender_export, profile="auto"):
    """Return JSON-serializable floor/counter/back/side planes and diagnostics.

    The source is a directory containing ``geometry.npz``. Direction signs are
    determined from the structural mesh: raised surface above floor, vertical
    faces toward floor's area-weighted centroid. ``auto`` requires exactly one
    complete object-name profile. Arbitrarily rotated and translated scenes are
    supported. Legacy role names floor/counter/back/side remain stable; each
    plane also has a display_name describing its actual surface.
    """
    path = Path(blender_export).expanduser().resolve() / "geometry.npz"
    with np.load(path, allow_pickle=False) as mesh:
        required = {"vertices", "triangles", "triangle_object_ids", "object_names"}
        if not required.issubset(mesh.files):
            raise ValueError(f"Target mesh lacks arrays: {sorted(required - set(mesh.files))}")
        xyz, ids, names, normals, areas = _triangles(mesh)
    profile_key, spec = _select_profile(profile, names, ids)
    patterns = spec["patterns"]
    extent = float(np.linalg.norm(np.ptp(xyz.reshape(-1, 3), axis=0)))
    tolerance = max(extent * 1e-6, 1e-7)
    groups = {name: _group(xyz, ids, names, normals, areas, rules, name)
              for name, rules in patterns.items()}
    up = groups["floor"]["normal"].copy()
    height_vector = groups["counter"]["centroid"] - groups["floor"]["centroid"]
    if abs(float(up @ height_vector)) < tolerance * 10:
        raise ValueError("Floor and raised surface do not establish a separated vertical direction")
    if up @ height_vector < 0:
        up *= -1
    if abs(float(groups["counter"]["normal"] @ up)) < .999:
        raise ValueError("Named raised surface is not parallel to the floor")
    planes = [_outer_face(groups[name], up, names, name, tolerance) for name in ("floor", "counter")]
    floor_center = np.asarray(planes[0]["centroid"])
    for name in ("back", "side"):
        direction = groups[name]["normal"].copy()
        if abs(float(direction @ up)) > .01:
            raise ValueError(f"Named {name} wall is not vertical relative to the floor")
        toward_room = floor_center - groups[name]["centroid"]
        sign = float(direction @ toward_room)
        if abs(sign) < tolerance * 10:
            raise ValueError(f"Floor footprint cannot determine the room-facing side of {name}")
        if sign < 0:
            direction *= -1
        planes.append(_outer_face(groups[name], direction, names, name, tolerance))
    for plane in planes:
        plane["display_name"] = spec["labels"][plane["name"]]
    n = np.asarray([plane["normal"] for plane in planes])
    d = np.asarray([plane["offset"] for plane in planes])
    if np.linalg.matrix_rank(n, tol=1e-6) != 3:
        raise ValueError("Structural surface normals do not span three dimensions")
    equations = np.column_stack((d, n))
    if np.linalg.matrix_rank(equations, tol=1e-6) != 4 or d[1] - d[0] <= tolerance * 10:
        raise ValueError("Structural planes do not constrain positive scale and translation")
    holdouts = []
    for name, (label, rules) in spec["holdouts"].items():
        if any(_matches(value, rules) for value in names[np.unique(ids)]):
            group = _group(xyz, ids, names, normals, areas, rules, name)
            plane = _outer_face(group, up, names, name, tolerance)
            plane["display_name"] = label
            holdouts.append(plane)
    return {"schema_version": 1, "profile": spec["identifier"], "planes": planes,
            "holdout_planes": holdouts,
            "diagnostics": {"mesh": str(path), "mesh_plane_tolerance": tolerance,
                            "profile_selection": {"requested": profile, "selected": profile_key},
                            "floor_counter_separation": float(d[1] - d[0]),
                            "floor_raised_surface_separation": float(d[1] - d[0]),
                            "normal_rank": int(np.linalg.matrix_rank(n)),
                            "offset_system_rank": int(np.linalg.matrix_rank(equations)),
                            "normal_singular_values": np.linalg.svd(n, compute_uv=False).tolist(),
                            "uses_camera_poses": False,
                            "profile_patterns": {k: list(v) for k, v in patterns.items()},
                            "limitations": ["Explicit names identify supported structural layouts; this is not generic object recognition.",
                                            "Only selected front/top mesh faces are anchors; unobserved extensions must not attract source points.",
                                            "Triangle winding is ignored; room-facing signs come from the floor footprint.",
                                            "Raised-surface-to-floor separation calibrates scale and is not held-out accuracy evidence."]}}
