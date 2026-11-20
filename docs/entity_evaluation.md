# IMG_1711 entity-mask evaluation

This experiment compares the VGGT-guided kitchen reconstruction with the original before-VQA reconstruction using their authored cameras and visible object silhouettes at the same **100 source timestamps**. It evaluates objects represented by each scene; it does not by itself measure whether every real object was reconstructed.

Open the [paired video comparison](../outputs/entity_comparison_1711/index.html). Both scenes have **100 completed full-resolution RGB/ID renders and completed SAM runs**. The [SAM comparison viewer](../outputs/entity_comparison_1711/sam_review/index.html) adds synchronized real-video masks, Blender category masks, and overlap diagnostics. Its scores compare category unions, not reviewed individual physical objects.

The imported runs contain 272 candidate tracks for the original scene and 273 for the guided scene. All 62 common concepts have identical text/anchor settings and byte-identical union masks across all 100 frames. Transfer auditing verified all 25,966 referenced PNG headers and sampled 2,160 decoded masks for binary encoding and recorded areas. See the [audit reports](../outputs/entity_comparison_1711/sam_audit/shared_concepts.json) and the per-scene candidate catalogs: [original](../outputs/entity_comparison_1711/before_vqa/sam_catalog/index.html), [guided](../outputs/entity_comparison_1711/vggt_guided/sam_catalog/index.html).

Each run has 23 categories with no detections. Empty model predictions are not verified physical absence. Visual inspection also found that the `microwave` prompt includes the countertop toaster oven, and `knife_block` includes a utensil crock. Raw union scores therefore remain diagnostic; they are not a final ranking of reconstruction quality. Candidate identities and sampled visual evidence are in the [identity review](../outputs/entity_comparison_1711/identity_review/index.html).

The [entity inventory](../outputs/entity_comparison_1711/entity_manifests.html) is a searchable, offline view of the exact proposed grouping. The machine-readable manifests are [before VQA](../configs/entities_1711_before_vqa.json) and [VGGT guided](../configs/entities_1711_vggt_guided.json).

| Scene | Authored mesh objects | Whole entities / surfaces | Unique SAM concepts | Groupings explicitly needing review |
|---|---:|---:|---:|---:|
| Original before VQA | 448 | 156 | 66 | 12 |
| VGGT guided | 290 | 164 | 67 | 10 |

Every exported authored mesh is assigned exactly once. A microwave body, door, controls and attached handle form one entity. Individual jars include their lids and labels; separate jars remain separate entities. Removable knives and dishes are separate from their holders/racks. Floorboards form one floor surface. Island stone top and waterfall end form one surface assembly, separate from the wooden base. Ambiguous appliance identities, loose cord ownership and uncertain small items remain visible entities with `needs_review` annotations.

These are name-based semantic proposals, not verified ground-truth labels. Entity IDs are positive integers local to a particular manifest; ID 0 is background. Matching IDs or similar names across scene manifests do not establish physical correspondence. Source `.blend` and object-export hashes bind each manifest to the exact inspected assets.

## Shared evaluation timestamps

Select 100 approximately uniform timestamps, retaining the first and last frame and each original global frame ID:

```bash
PYTHONPATH=src python3 -m scene_reconstruction select-evaluation-frames \
  --manifest data/frames/manifest.json --count 100 \
  --output configs/evaluation/1711_frames_100.json
```

This is approximately **4.26 selected intervals per second** across the 23.235-second first-to-last timestamp span, not 2 fps. The images keep the canonical upright 1080×1920 grid. Every stage joins by original frame ID and exact timestamp rather than treating selected frames as a new consecutive 30-fps sequence.

The saved selection file is an additional subset manifest. The canonical `data/frames/manifest.json` remains the identity and provenance reference for all 698 video frames. Existing output paths are protected; use a new destination to repeat an experiment without mixing results.

## Render RGB and entity IDs

On this Mac, use Metal for the full-resolution eight-sample Cycles render. Both scenes use the same source timestamps and rendering settings:

```bash
PYTHONPATH=src python3 -m scene_reconstruction render-entities \
  --blend blender_scenes/before_vqa/scene.blend \
  --manifest data/frames/manifest.json \
  --selection configs/evaluation/1711_frames_100.json \
  --entities configs/entities_1711_before_vqa.json \
  --render-device METAL --samples 8 --resolution-scale 1 \
  --output outputs/entity_comparison_1711/before_vqa/render

PYTHONPATH=src python3 -m scene_reconstruction render-entities \
  --blend blender_scenes/vggt_guided_1711/scene.blend \
  --manifest data/frames/manifest.json \
  --selection configs/evaluation/1711_frames_100.json \
  --entities configs/entities_1711_vggt_guided.json \
  --render-device METAL --samples 8 --resolution-scale 1 \
  --output outputs/entity_comparison_1711/vggt_guided/render
```

`--frames 0` can restrict a new output directory to a smoke render. `--no-rgb` skips the authored-color render. `--render-device CPU`, `CUDA` or `OPTIX` selects other supported execution devices; availability is validated rather than inferred. The original `.blend` files are opened without saving over them.

The mask pass uses Blender's **object-index pass**, assigning the same positive ID to all mesh parts of an entity. It records depth-occluded visible labels rather than hidden/amodal silhouettes. This is a hard label image, not a Cryptomatte coverage matte: edge anti-aliasing, transparent materials, motion blur and shallow depth of field can differ from the real video's image formation. Read the render's recorded settings and mask semantics before interpreting a boundary discrepancy.

Keep the authored camera trajectory, focal length and scene geometry fixed for this comparison. Do not apply the VGGT-to-Blender surface alignment or optimize a separate image alignment before scoring: that would change the reconstruction being evaluated. Blender frames are evaluated from original video time with subframes, not by renumbering the 100 selected images.

Render completion is recorded in each directory's `index.json`. Both local runs are complete and all 200 RGB/ID rasters passed checks for grid, label IDs, timing, camera matrices and source hashes. Synthetic CPU and Metal checks verified occlusion, multipart grouping, image orientation and alpha-threshold behavior. All 156 unguided entities appear at least once; two guided entities (`upper_bridge_cabinet`, `thin_dark_bottle`) have no visible pixels at any selected timestamp and retain empty masks. Eight-sample RGB is an inspection preview, not a photometric-quality benchmark. Raw `frames/*.npz` arrays contain the authoritative `uint16` `entity_ids`; binary PNG export is optional until entity identities are associated.

## Prepare and run scene-specific SAM

After both complete entity renders, create shared text/anchor settings so a common concept receives identical inputs in both batches:

```bash
PYTHONPATH=src python3 -m scene_reconstruction shared-entity-prompts \
  --selection configs/evaluation/1711_frames_100.json \
  --entities configs/entities_1711_before_vqa.json configs/entities_1711_vggt_guided.json \
  --render-indices outputs/entity_comparison_1711/before_vqa/render/index.json outputs/entity_comparison_1711/vggt_guided/render/index.json \
  --output configs/evaluation/1711_shared_prompts.json
```

The completed shared file contains 71 concepts, of which 62 occur in both inventories. It chooses the timestamp with the largest summed visible concept area across both renders, treating the scenes symmetrically and choosing the earlier frame on ties. This controls only the text-prompt timestamp; it does not supply scene masks, boxes or points to SAM. Independent SAM runs can still differ, so shared-entity scoring should use consistently reviewed physical-object references.

Then prepare each scene's sparse source-video inputs:

```bash
PYTHONPATH=src python3 -m scene_reconstruction prepare-entity-sam \
  --manifest data/frames/manifest.json \
  --selection configs/evaluation/1711_frames_100.json \
  --entities configs/entities_1711_before_vqa.json \
  --render-index outputs/entity_comparison_1711/before_vqa/render/index.json \
  --anchor-prompts configs/evaluation/1711_shared_prompts.json \
  --output outputs/entity_comparison_1711/before_vqa/sam_inputs

PYTHONPATH=src python3 -m scene_reconstruction prepare-entity-sam \
  --manifest data/frames/manifest.json \
  --selection configs/evaluation/1711_frames_100.json \
  --entities configs/entities_1711_vggt_guided.json \
  --render-index outputs/entity_comparison_1711/vggt_guided/render/index.json \
  --anchor-prompts configs/evaluation/1711_shared_prompts.json \
  --output outputs/entity_comparison_1711/vggt_guided/sam_inputs
```

The prepared `manifest.json` references the original video frames using portable relative paths. `prompts.json` contains one text concept per unique category in that scene; `entity_candidates.json` preserves all physical entities awaiting association. The optional render index only selects a prompt timestamp with high predicted visibility. No Blender boxes, points or masks are supplied to SAM. Omit `--render-index` to prepare before renders finish; this uses the first selected frame as the prompt anchor and records that choice.

The two `sam_inputs/` folders are already prepared locally. Transfer the repository code/configs and canonical frames together with them, preserving relative layout. The generated `outputs/` folders are Git-ignored, so pushing the code alone does not transfer these prepared inputs. After pushing/pulling code and configs, run this on the Mac from the repository root:

```bash
rsync -avP --relative \
  outputs/entity_comparison_1711/before_vqa/sam_inputs/ \
  outputs/entity_comparison_1711/vggt_guided/sam_inputs/ \
  oceanusm@pluslab02-a6000.cs.ucla.edu:/local1/oceanusm/img1711_reconstruction/
```

The remote already used `data/frames/`; keep those exact canonical frames and manifest. Blender RGB/ID renders are not needed on the remote for SAM inference. SAM runs on the A6000 with the environment and authorized checkpoint described in [SAM3 setup](sam3.md). From the remote repository root:

```bash
CUDA_VISIBLE_DEVICES=7 bash scripts/run_entity_sam.sh \
  outputs/entity_comparison_1711/before_vqa/sam_inputs \
  outputs/entity_comparison_1711/before_vqa/sam

CUDA_VISIBLE_DEVICES=7 bash scripts/run_entity_sam.sh \
  outputs/entity_comparison_1711/vggt_guided/sam_inputs \
  outputs/entity_comparison_1711/vggt_guided/sam
```

Append `--dry-run` to validate transferred inputs and print the invocation without starting inference. The default is one continuous 100-selected-frame tracking session **per concept**, saving instance masks. “Once per scene” means one scene-specific batch, not that SAM processes every text category in a single model call. There are 66 and 67 concepts in the proposed inventories; cost depends on concept count and detected instances as well as frame count. `SAM_CHUNK_SIZE` can bound sequence length, but resets tracking identities at chunk boundaries. Sparse input is lighter but its larger temporal jumps can weaken tracking.

## Inspect imported SAM results

After pulling both `sam/` directories, generate the paired category diagnostic locally:

```bash
PYTHONPATH=src python3 -m scene_reconstruction review-entity-sam \
  --root . --preview-height 480 \
  --output outputs/entity_comparison_1711/sam_review
```

The command validates source manifests, timestamps and render provenance, then merges Blender entities by their `concept`. It compares each resulting visible category union with the corresponding SAM union at the original 1080×1920 resolution. Only browser previews are reduced. It records per-frame intersection, union, IoU and Dice, and checks whether the shared SAM references agree across runs. Both-empty masks have undefined IoU rather than receiving a perfect score. A zero SAM mask is flagged as unresolved detection/absence, not automatically accepted as correct absence.

The viewer synchronizes concept and source time across both scenes. Magenta shows SAM, cyan shows Blender, and white shows overlap. The authored Blender cameras and object positions remain unchanged. These diagnostics can reveal pose, silhouette and timing differences, but different per-scene inventories and SAM category errors preclude treating their aggregate scores as a final guided-versus-unguided benchmark.

For the individual-instance review, generate each candidate catalog if needed:

```bash
for scene in before_vqa vggt_guided; do
  PYTHONPATH=src python3 -m scene_reconstruction catalog-instances \
    --manifest "outputs/entity_comparison_1711/$scene/sam_inputs/manifest.json" \
    --sam-output "outputs/entity_comparison_1711/$scene/sam" \
    --output "outputs/entity_comparison_1711/$scene/sam_catalog"
done
```

These output directories must be new or empty; the imported experiment already has both catalogs. Use a new output path when regenerating.

## Associate instances before scoring

A prompt such as `bottle` may detect many bottles, including real objects absent from the Blender scene. That concept union cannot serve as the individual reference mask of every Blender bottle. Both unique and repeated scene entities require a checked semantic association to SAM tracks. A tracker identity is `(concept, chunk_start_index, instance_id)`; the integer ID alone is not globally meaningful.

Use `catalog-instances` to inspect saved SAM tracks and `assemble-instances` with an explicit reviewed mapping to build per-entity masks. Both accept the prepared sparse manifest while retaining canonical IDs. Never choose the SAM instance with maximum overlap against each Blender mask and then score the same frames: that can favor whichever reconstruction happens to cover a different real item.

Maintain a shared physical-entity association across the two scene inventories where the original video supports one. Report at least:

- **Shared entities:** the same real objects, timestamps and reference masks for a direct guided-versus-unguided comparison.
- **Each scene's represented inventory:** scene-conditional results, including unmatched and ambiguous targets rather than silently dropping failures.
- **Segmentation review status:** absent detections, identity switches, uncertain groups and masks excluded for documented reasons.

Concept-level unions are a useful additional diagnostic when compared with the matching Blender entity union, but answer a different question from per-object alignment. Scores from inventories with different difficulty and coverage are not directly interchangeable.

## Export masks and inspect video alignment

Once the render and prepared SAM manifest are complete, export binary Blender entity masks with matching sparse-manifest provenance:

```bash
PYTHONPATH=src python3 -m scene_reconstruction export-entity-masks \
  --render-index outputs/entity_comparison_1711/before_vqa/render/index.json \
  --entities configs/entities_1711_before_vqa.json \
  --manifest outputs/entity_comparison_1711/before_vqa/sam_inputs/manifest.json \
  --output outputs/entity_comparison_1711/before_vqa/blender_masks

PYTHONPATH=src python3 -m scene_reconstruction preview-entities \
  --manifest data/frames/manifest.json \
  --render-index outputs/entity_comparison_1711/before_vqa/render/index.json \
  --entities configs/entities_1711_before_vqa.json \
  --output outputs/entity_comparison_1711/before_vqa/preview
```

Repeat with `vggt_guided` output paths and `configs/entities_1711_vggt_guided.json` for the guided scene. The preview writes `preview/index.html` and `preview/comparison.mp4`. Before SAM, the Blender entity-ID tints are available alongside the timestamped source and rendered images. Once per-entity SAM tracks have been reviewed and assembled, supply `--sam-masks PATH/TO/assembled/masks --sam-manifest PATH/TO/sam_inputs/manifest.json` to add their image-coordinate overlays in a new preview output directory.

For a reviewed set of corresponding entity names, the existing comparison command scores the assembled masks against the exported Blender masks:

```bash
PYTHONPATH=src python3 -m scene_reconstruction compare-masks \
  --manifest outputs/entity_comparison_1711/before_vqa/sam_inputs/manifest.json \
  --predicted outputs/entity_comparison_1711/before_vqa/assembled/masks \
  --reference outputs/entity_comparison_1711/before_vqa/blender_masks/masks \
  --objects range microwave \
  --output outputs/entity_comparison_1711/before_vqa/mask_scores
```

`range microwave` is an example of an explicit reviewed subset, not an assertion that these mappings have already been reviewed. Use the same confirmed physical targets and frame exclusions for the paired guided-versus-unguided result. A complete binary export includes an all-zero mask when an entity is absent; missing files indicate incomplete data.

The review pairs original video RGB with authored Blender RGB at the same timestamp. Blender ID tints show what the scene claims is present; SAM tints show the corresponding detected real-object masks. A SAM mask over the Blender render is an image-coordinate overlay, not a reprojected geometric observation. A mismatch can come from camera pose/intrinsics, geometry, object motion, occlusion, timing, grouping or segmentation error.

The source includes a moving hand and transferred container. A static model may correctly represent an object while missing its motion or occlusion. Keep those frames visible in the review; only apply explicitly reviewed ignore regions or static-interval filters, and report excluded coverage separately. Do not infer invisible objects from missing mask files: complete per-entity outputs require zero masks when absent and an error when data is missing.

Iterative SAM prompting and frame review can improve inventory coverage, but cannot guarantee completeness. Small, transparent, heavily occluded or never-observed objects can remain undetected. The present experiment deliberately measures video alignment for represented objects; broader completeness evaluation is a separate stage.
