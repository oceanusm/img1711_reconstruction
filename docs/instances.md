# Curating every distinct item

SAM3 text concepts produce candidate instances, not a complete inventory of
physical objects. One broad prompt may miss unusual items; several prompts may
find the same object. Objects may also split into several tracks, and tracker IDs
restart in each chunk. The curation tools turn reviewed candidate tracks into
stable names such as `mug_01`, `mug_02`, `floor`, and `countertop` for comparison
with Blender. They do not infer that two tracks depict the same physical item.

Run SAM3 with instance exports enabled, then make the candidate catalog:

```bash
scene-recon catalog-instances --sam-output outputs/sam --output outputs/instance_catalog
```

Open `outputs/instance_catalog/index.html`. Each row contains a representative
source frame with its predicted mask highlighted, a track reference, the chunk
range, and the frames on which that track was returned. `catalog.json` contains
the same inventory in machine-readable form. It is a review aid, not proof that
all objects were discovered. Review the source video throughout its duration,
including items newly revealed by camera motion or by moving a container.

Copy `mapping.template.json` to a new file and fill its `objects` list. Each
physical item or desired surface gets one stable output name. A track reference
is the triple `(concept, chunk_start_index, id)` shown in the catalog. Merge
several references into one output object when they are the same physical item
across chunks or fragments of that item. Review different concepts carefully:
`cup`, `container`, and `object` can all detect the same vessel.

For a hypothetical 240-frame video with 120-frame chunks:

```json
{
  "schema_version": 1,
  "objects": [
    {
      "name": "jar_01",
      "tracks": [
        {"concept": "containers", "chunk_start_index": 0, "id": 3},
        {"concept": "jars", "chunk_start_index": 120, "id": 1}
      ],
      "absent_ranges": [[48, 63]],
      "missing_detections": "error"
    }
  ]
}
```

All range endpoints are inclusive global source-frame indices. A track applies
throughout its source chunk by default; add `start_frame` and `end_frame` to use
only part of that chunk, for example to exclude a tracker identity switch or
false-positive segment. You may select the same track several times for the
same object with disjoint ranges. A track cannot be assigned to two different
output objects. References from different concepts can still describe duplicate
physical items; the tool cannot determine that identity automatically.

Every frame needs a deliberate disposition:

- Selected track ranges cover frames expected to have model evidence.
- `absent_ranges` declares reviewed frames where that item has no visible mask,
  including full occlusion or leaving the image. A selected nonempty mask on
  such a frame is a conflict and stops assembly; narrow the track range when
  discarding a false-positive segment.
- If a selected track is not returned on a covered frame, the default
  `missing_detections: "error"` stops assembly. Inspect the gap and add an
  absence range or repair/re-run the segmentation.
- Only after reviewing those gaps, `missing_detections: "reviewed_absent"`
  explicitly authorizes empty masks when no selected track is returned. Those
  frames are marked accordingly in the output. This does not authorize missing
  files or uncovered chunks; both still fail.

An entirely unassigned chunk is unknown, not an empty mask. It must receive a
track assignment or an explicit absence range. The tool does not convert
segmentation failure into verified object absence.

Assemble the reviewed mapping:

```bash
scene-recon assemble-instances \
  --sam-output outputs/sam \
  --mapping configs/instance_mapping.json \
  --output outputs/objects
```

The source SAM run must be complete and match the exact frame manifest hash.
Every selected instance needs its saved binary PNG; rerun SAM with
`--save-instances` if those were omitted. Source dimensions are enforced without
resizing masks. Assembly writes:

```text
outputs/objects/
  metadata.json
  frames.jsonl
  overlaps.json
  masks/<stable_object_name>/<global_frame_index:06d>.png
```

`frames.jsonl` records selected tracks and whether the mask came from tracking,
a returned empty model mask, explicit manual absence, or the reviewed-gap
policy. `metadata.json` includes the comparator-compatible `manifest_sha256`
field (also retained as `source_manifest_sha256`) and hashes of the mapping and SAM
metadata/index. `overlaps.json` reports pixels assigned to several curated
objects; inspect these for duplicate identities or segmentation errors. These
are independent binary object masks, not a panoptic label image: overlapping
pixels are reported and preserved.

Use `--predicted outputs/objects/masks` in `compare-masks`, with matching Blender
object names. Separate dynamic objects from static reconstruction constraints.
Successful assembly confirms the declared mapping and frame coverage; it does
not certify that the inventory is complete or that its masks are correct.
