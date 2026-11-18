# Segment the scene at the level of whole items

The target is a mask for each distinct visible item, together with useful scene surfaces, at a level that can be matched to Blender objects. A bottle should normally include its cap and label; a stove should include its knobs and door; a potted plant should include its leaves and pot. A separate jar beside the bottle is a separate item. Texture patches, printed words, individual buttons, and material changes should not automatically become scene objects.

`configs/objects.json` now contains 30 initial concepts, based on the video contact sheet and several full-resolution frames. It covers the floor, walls, worktops, island base, cabinets, shelves, window, appliances, cookware, dishes, tools, containers, packets, plants, and small counter items. A single concept such as `bottle` can return many separate instances. The configuration is a starting inventory to review after inference; no masks have been accepted as complete merely because their class name appears in this file.

## SAM 3 has two relevant prompting tasks

| Task | What identifies the target | Intended result | Use in this project |
| --- | --- | --- | --- |
| Promptable Concept Segmentation (PCS) | A short noun phrase, an image exemplar, or both | Instances matching that concept | Find all bottles, jars, appliances, and other listed categories |
| Promptable Visual Segmentation (PVS) | Points, a box, or an input mask identifying a particular object | That selected object's mask, with video tracking where supported | Correct or recover a specific missed jar or an ambiguous surface |

These are the task definitions in the [SAM 3 paper](https://arxiv.org/html/2511.16719v2). A visual box can serve different purposes: an exemplar box asks for objects of the demonstrated concept, while an instance-selection box identifies one target. The intended API path matters; a box alone is not a promise that the output will contain exactly one instance. Meta describes the [text and exemplar prompting behavior](https://ai.meta.com/sam3/) separately.

The current `scene-recon sam` command implements **text-prompted video PCS**. It exports concept-union masks and, by default, individual instance masks. It does not provide an interactive clicking interface, a box/mask prompt file format, or automatic PVS correction. The official repository supplies examples for [video text prompting and point refinement](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_video_predictor_example.ipynb) and [image text/box prompting](https://github.com/facebookresearch/sam3/blob/main/examples/sam3_image_predictor_example.ipynb); integrating those interactions would be additional work.

## Prompt modes are different from output organization

The distinctions below describe how masks are represented, not extra switches that make SAM discover the entire scene:

| Representation | Example | What the current pipeline has |
| --- | --- | --- |
| Semantic/category mask | All bottle pixels share one label | One binary union per prompted concept and frame |
| Instance masks | Bottle A and bottle B have different IDs | Individual masks and session-local track IDs |
| Panoptic scene labeling | Separate object IDs plus surface classes, with one accepted label per pixel | Not produced; concepts can overlap and some pixels can be unlabeled |

For example, combining every jar into one mask gives useful category coverage but loses the distinction between their positions. Per-item Blender comparison therefore needs the instance outputs and a reviewed instance-to-Blender mapping. A scene-wide panoptic result would additionally require resolving overlaps, duplicate detections, boundaries between surfaces, and gaps. It should preserve an explicit unknown/void region where evidence is insufficient rather than inventing a confident label.

There is no reliable `everything` text prompt that guarantees one correct mask for every object and surface. Prompting every pixel or requesting many generic regions can also produce overlapping parts, duplicate objects, and arbitrary granularity. Here, a reviewed scene inventory defines what counts as an item; model masks propose its visible extent. Fine detail, transparency, reflections, unfamiliar categories, and visually ambiguous object boundaries can still require correction. See the authors' discussion in Appendix B of the [SAM 3 paper](https://arxiv.org/html/2511.16719v2).

## Grouping choices for this kitchen

The `kind` and `grouping_policy` fields in the configuration explain the intended evaluation units. The adapter currently ignores these fields during inference; only `text` and the anchor frame guide the model. They are review instructions, not enforced mask-merging rules.

- Keep caps, fitted lids, handles, labels, and attached controls with the parent object. A lid lying separately on a counter may instead be its own loose item.
- Preserve distinct removable tools, jars, bottles, plates, and packets. Several objects sharing a prompt should still have separate instance masks.
- The knife block and utensil holders are separate from their removable tools. A knife's handle and blade remain one knife, even when only the handle is visible.
- Separate worktop surfaces from their supporting island/cabinet bodies. Do not also add full-island masks into the same flat pixel label set without resolving the overlap.
- Keep wall/backsplash and floor regions at a useful surface scale. Floorboards, reflections, grout lines, and cast shadows are not separate objects by default.
- Grouping a plant with its pot, or a paper roll with its holder, is an explicit practical choice. The same grouping must be used on the Blender side.

Some text concepts naturally overlap. A tub may also be detected as a jar; `food package` may recover a bottle already found elsewhere; `dishware` may include cookware. Review these detections across prompts and retain one accepted identity per physical item. Do not resolve conflicts by arbitrary prompt order. Shorter, more specific replacement phrases or future visual corrections can help once a failure is observed.

## Review and assign identities before measuring Blender agreement

1. Inspect several views across the full clip, especially close-ups of both shelves and the right countertop. Add any missed physical items to the inventory. Check the configuration's `inventory_review_candidates`, including small packets, loose coffee capsules, trays, cords, outlets, transparent wrapping, and a faucet if visible.
2. Run the text inventory with instance saving enabled. Review per-instance overlays for missed items, merged neighbors, detached parts, false positives, and category duplicates. A missing detection remains a missing detection; a black mask alone cannot distinguish absence from model failure.
3. Create a persistent scene inventory with an ID such as `shelf_lower_bottle_03`, a human description, its accepted grouping, and its corresponding Blender object or object group. Map accepted SAM tracks to those IDs. No automatic global identity assignment or Blender matching is currently implemented.
4. Inspect identity continuity at every chunk boundary and after occlusion. Current inference starts independent chunk sessions, so `(concept, chunk_start_index, id)` identifies a local track. The same numeric object ID in two chunks does not establish the same physical item. Merge and split decisions need review, especially for similar jars that move.
5. Export complete binary visible masks on the canonical frame grid for each accepted scene ID. Use these masks for per-item comparison, and keep concept unions for category coverage checks. Record the reviewed mapping and any exclusions alongside the results.

The 30 concepts are a practical initial batch, not an upper limit on scene IDs: one `bottle` query can cover a dozen distinct bottles. More prompts increase runtime because the current adapter runs each concept separately. Add targeted concepts for observed gaps instead of permanently accumulating many synonyms for the same items.

## Visibility, motion, and hidden surfaces

The clip contains a moving hand/forearm and moved containers. A visible mask excludes pixels covered by the hand or another object. Rotation also changes which parts of an object hide other parts of itself. The mask should not fill an imagined complete silhouette behind an occluder.

For fair Blender overlap, render the same visible portions at the same timestamps, with matching object motion and occlusion. If an occluding hand is intentionally absent from Blender, pass `--ignore-masks outputs/sam/masks/hand` to `scene-recon compare-masks`. The comparator records the ignored-pixel count and separately reports scores that exclude those pixels. Review this dynamic mask first: the hand/forearm prompt may miss sleeve pixels or wrongly include nearby objects. Otherwise a correct static object can receive a poor overlap score for a visibility mismatch.

Masks alone cannot reconstruct the hidden back of an object or prove metric distance. Their value is strongest when they are combined with camera alignment, depth, multi-view consistency, and the observed visibility history. Keep unreviewed SAM proposals separate from accepted evaluation masks so model errors are not mistaken for errors in the Blender scene.
