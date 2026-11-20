# img_1696 — interior reconstruction

## Using the files

Open **scene.blend** for measurement or scene-only VQA. Select a mesh or curve in one of the five numbered collections to inspect a physical part. The camera opens on source frame 0. Advancing Blender's timeline follows the original pan through the dining area, seating, TV wall, bar, kitchen, console, and back to the windows.

**scene_with_vggt.blend** contains the same authored scene plus `VGGT_REFERENCE`. The original 5,000,000-vertex mesh retains its native XYZ, byte RGB, per-point confidence and source-frame index. Its object transform places it in the metric scene. A visible 1-in-50 sample makes interactive comparison practical. Toggle visibility in the Outliner to compare against the complete cloud. The collection is disabled for rendering. The clean scene, object manifest, measurements and movie exclude it.

Use **manifest.json** to look up Blender object names, component relationships, geometry-derived dimensions and bounding boxes. **camera.json** has one evaluated camera-to-world matrix and one pixel calibration matrix for each oriented source frame. **vggt_alignment.json** contains source hashes, conventions, the exact similarity transform, the 450 reference-frame mappings and discrepancy statistics. **inspection/** contains source and reference comparisons for auditing. Keep these inspection files separate from scene-only VQA inputs.

The manifest distinguishes component bounds from assembly bounds. `bbox_m` is the minimum and maximum of evaluated vertices in world XYZ on source frame 0, including curve tessellation, solidification and bevels. `dimensions_m` is the corresponding extent along the object's modeled local axes. Meshes authored directly in world coordinates use room axes. Parent entries may also list `assembly_bbox_m`, which includes their child components. First-observation frames are estimates of when an item becomes identifiable. For later observations, the manifest includes a second box evaluated on the recorded source frame. Physical properties and material identities are estimates unless their visible features establish them.

## Coordinate frame and metric scale

One Blender unit represents one estimated metre. The floor is the XY plane. +X points from the long artwork wall toward the TV, +Y runs toward the end windows, and +Z points up. The origin lies at the intersection of the fitted floor and artwork-wall datums, in the transverse plane through the first reference camera.

The original VGGT scene has arbitrary scale. The scale anchor is a **0.75 m assumed dining-table height**. Floor and wall planes were initialized with robust plane fits and refined with trimmed least squares. Five independent tabletop clicks in the opening view supplied floor clearances. The resulting similarity uses **5.461832 m per VGGT unit**. It applies the same scale, orthonormal rotation and translation to every reference vertex and camera center. Camera orientations use the rotation alone.

There is no tape measurement, surveyed control point or confirmed furniture model. A true table height of 0.72–0.78 m would change every length by approximately −4% to +4%. Two independent views of the duplex outlet plate yielded about 72 × 112 mm, a plausibility check on the scale. The installed plate model is unknown. A small residual in this project indicates close agreement with the source reconstruction at the assumed scale. It does not establish survey accuracy.

## Evidence and scene authoring

The MOV contains 992 displayed portrait frames over 33.0683 seconds. The matching VGGT index references 450 of them and describes an unfused cloud sampled from predicted depth. The reconstruction used the time labels on the points to compare each projected feature with its corresponding video view. The full cloud was loaded into native Blender before the model was finalized. Intermediate dining and console assemblies were saved with it, and the integrated scene was compared against its surfaces before the final render.

The floor continues under the furnishings. The end wall has separate piers, real window openings, frames, panes, meeting rails and roller shades. The media wall has a ribbed profile, a recessed firebox, crystal bed, glazing, separate TV body and display, and a hanging lead. The dining section includes six wishbone chairs, separate woven seats, gray textile, open carved bowl, runner, pendant and framed print. The seating includes sectional modules, loose cushions, bolster, two visible draped layers, rug, low table, closed laptop, gamepads, remotes, candles, book, can and smaller items. The window armchair, turned end table, ceramic lamp, tall floor lamp, wall outlets, plants, bar cart, bottles, stemware, console, games, vases, flowers and close-up accessories are editable. The kitchen and the street visible through the glazing have their own solid geometry.

Major planar surfaces use fitted geometry. Slats, edges, bevels, chair frames, turned components, stamped metal, cushion shapes and plant leaflets use object-specific meshes. The throw surfaces use traced masks and robust local 3D depth before thickness, contour smoothing and contact corrections. Some finishes use closely cropped and rectified object pixels, including the botanical panel and textile swatches. The TV material retains isolated architecture from the photographed screen and reconstructs its skyline, weather tile, empty apertures and street. No source frame is used as a background or composited into the movie. Printed portraits, onscreen figures and real people were excluded. The bar mirror reflects only the modeled scene.

### Coverage, uncertainty and deliberate departures

| Region | Evidence and modeling decision |
|---|---|
| Floor, dining table, art wall, console and coffee table | Dense support on exposed faces. Modeled surfaces remain planar across noisy points and holes. The tabletop yaw and peninsula edge were revised using the front and reverse views. Concealed undersides and joinery are estimates. |
| Sectional, throw and chairs | Accessible seat tops and fabric folds are covered. The near chaise edge projects past its arm. Window-side cushions lean back in accordance with depth versus height. The seat backs occlude their bases. A pale upper throw layer may be a separate blanket or the reverse of the taupe layer. |
| Windows and ceiling | Opaque shades, glare, glass and blank ceiling produce few returns. Window edges and sash heights were located from image rays and solid piers. Walls and glazing continue through missing areas. Ceiling light positions, internal framing and construction thickness carry lower confidence. |
| Media wall | The display is flat and the firebox has a real recess. The source cloud contains screen-depth scatter and few reliable points inside the hearth. Emissive flame patterns and TV media are human-free approximations. Hidden mounting hardware and wiring are unresolved. |
| Bar and console accessories | Major shelf and canvas planes are well supported. Thin glass, stems and bottle boundaries are less stable. Some labels, tiny ornaments, contents of containers and fine printed matter are simplified. Small bead-like items and the textile below the coffee table have uncertain identities. |
| Kitchen and through-window exterior | Brief, distant views constrain visible silhouettes. Counter and appliance depths, interior carpentry and small clutter remain estimates. The parked red vehicle is located by parallax through two windows. Road elevation, other vehicles, roofs, vegetation, far facades and concealed faces have much weaker depth evidence. Geometry outside the visible bounds is approximate. |

## Camera, motion and timing

The camera uses the display-oriented 1080 × 1920 image, local +X right, +Y up and −Z forward. The reference cameras use OpenCV axes. The supplied basis change flips local Y and Z. The pose matrices in camera.json map homogeneous column vectors from the final Blender camera into the metric world. The square-pixel focal estimate uses the geometric mean of the original horizontal and vertical pixel focal lengths. Distortion was not independently calibrated.

All 450 reference poses are retained at their timestamps. At 323 other frames, tracked image features and depth supported a local pose refinement. The remaining 219 frames use timestamp-weighted interpolation. Refined intermediate poses change by a median of 6.7 mm and 0.18° relative to interpolation. Across 36,246 held-out feature correspondences in the intermediate views, the median reprojection discrepancy fell from 2.55 to 1.84 displayed pixels and the 90th percentile from 6.53 to 6.39 pixels. Adjacent-reference tracks had a 5.54-pixel median and 17.57-pixel 90th percentile. These diagnostics use geometry and camera evidence from the same reconstruction.

No independent rigid furniture or object motion could be identified in the pan. The chairs, laptop lid, shades and cabinet doors stay in their observed configurations. No joint travel is inferred from a single stationary configuration. The fireplace has animated material illumination, including spatially varying flame texture and ember brightness. Fine media timing cannot be recovered from the short TV and fireplace passages. Exterior vehicles remain parked.

Blender frame 1 is source frame 0. The scene runs at nominal 30 fps. The MOV's actual frame durations vary by a small number of 1/600-second ticks. The render encoder restores the source PTS, container display orientation and duration after rendering the upright views.


## Dimensions and geometric checks

Selected estimates below are rounded to communicate their measurement basis. The solid geometry retains additional digits for reproducible comparisons.

| Feature | Reconstructed estimate | Basis |
|---|---:|---|
| Dining top | 1.74 m long × 0.86 m wide | Multiple corners and tabletop returns. Its 0.75 m top elevation sets global scale. |
| Coffee tabletop | 1.30 m long × 0.71 m wide | Perimeter rays intersected with its supported top plane. |
| Rectangular rug | 2.95 m × 2.39 m | Exposed rug edges projected onto the floor. Covered areas and the far corners are inferred. |
| Mounted television chassis | 1.68 m wide × 0.96 m high | Display boundary and colored surface points. Chassis depth is less certain. |
| Mantel-free electric fireplace | About 2.42 m clear width | Ends of the long dark surround above the crystal bed. |
| Space across the living area | About 4.6 m from artwork-wall datum to TV cladding | The cladding is slightly angled relative to the room axes. The width varies along the room. |

Distances below run from a deterministic sample of source PLY vertices to the nearest triangle of the evaluated authored geometry at Blender frame 1. The calculation includes bevels and curve tessellation and omits the reference objects and render aids. These are one-sided internal fit statistics. Repeated observations, outliers, poor coverage and a nearby unrelated surface can affect them. They do not bound errors on unobserved faces.

| Region | Reference samples | Median distance | 90th percentile |
|---|---:|---:|---:|
| All sampled points | 60,241 | 7.9 mm | 33.7 mm |
| Exposed floor walkway | 497 | 10.4 mm | 15.8 mm |
| Long artwork wall | 1,116 | 10.2 mm | 27.4 mm |
| Window-wall pier and artwork | 2,285 | 7.3 mm | 32.0 mm |
| Dining top and runner | 1,218 | 1.7 mm | 5.5 mm |
| Coffee table and contents | 1,203 | 4.8 mm | 13.3 mm |
| Television display | 1,386 | 13.7 mm | 35.2 mm |
| Media-wall strip above hearth | 676 | 6.4 mm | 55.1 mm |
| Chaise seating | 1,705 | 17.1 mm | 50.1 mm |

The largest well-sampled peninsula cell improved from a median 256 mm discrepancy to 2.4 mm. Two chaise-front cells fell from 103–119 mm to 8.6–9.7 mm. The lampshade region improved from 55 mm to 5.7 mm. The before/after cell selections are retained in `vggt_alignment.json`.

The first integrated fit exposed a missing section of the peninsula, a recessed chaise cushion, excessive window-side cushion depth and a displaced lamp. The peninsula front was extended to the projected opening-view corner. The chaise cushion was extended approximately 0.14 m beyond its arm. Two window-side back cushions were tilted to follow supported depths, and the lamp was positioned toward the rear of its tabletop. Other revisions corrected the window armchair silhouette, outlet height, framed botanical print, cloth boundaries and canvas depth. Small, unsupported details retain lower confidence.

Separate, explicitly bounded plane checks returned median absolute discrepancies of 3.5 mm on the exposed long-console top, 7–8 mm on the relief canvas faces, and 12–13 mm on the bar cart shelves. The script `check_decor.py` documents the point-selection windows. The fitted peninsula corner returned a 2.2 mm median height discrepancy across 618 selected points. The table, shelf and wall checks measure a different selection than the whole-scene audit.

In 14 manually marked dining-table corners across four camera views, the median discrepancy was 9.6 pixels and the 90th percentile was 17.0 pixels in the 1080 × 1920 image. The corner observations contributed to the fit.

## Delivered-file validation

Blender 4.2.23 LTS reopened the clean file. Its 482 geometric objects match the manifest by exact object name. All 14 file-backed textures are packed. A second implementation independently transformed each evaluated vertex and checked 872 initial or first-observation bounding boxes. The largest coordinate difference was 1.5e-06 m, a numerical consistency result. All 478,144 per-frame object-transform comparisons passed. The tracked camera and intrinsics were evaluated again on all 992 frames. Spot checks of eight flat-base contacts, including the can and laptop on the coffee table, the stacked games and the table lamp, found base-to-support height differences within 3 mm. An `on` relationship can name a supporting assembly, such as the chaise or the drinks trolley; its highest vertex may be on a separate rail or cushion. See `inspection/09_scene_validation.json` for the numerical checks.

The inspection file also reopened with 5,000,000 full-resolution source vertices and colors in its render-disabled collection. A deterministic 113-point comparison recovered the original XYZ coordinates exactly and byte RGB within rounding tolerance. The aligned object matrix agreed with the documented transform to 0.0000003 per element; the 100,000-point inspection sample was visible. Six inspection plates show the room plan, artwork elevation, media wall, window elevation, oblique cutaway and spatial distribution of point-to-model error. The source/render contact sheet covers the entrance, sweeps, close-ups and final view.

The motion diagnostic in `inspection/08_image_motion.json` compares original and rendered apparent displacement. It found sufficient features in 983 of 991 adjacent-frame pairs. Across 14,219 matched image regions, the median difference was 2.34 pixels and the 90th percentile was 8.91 pixels. The largest differences occurred on the approach to the TV near frames 398–408 and at the outdoor window around 846–847. Feature choice, the approximated media graphic and simplified exterior contribute to this image-space error.

The final movie was rendered from the clean scene with Blender EEVEE at eight temporal samples. The saved Blender scenes use a 12-sample render default. The encoding stage decoded the encoded output, verified 992 frames, compared frame timestamps and checked stored size, display rotation, time base and duration against the MOV. The portrait image is 1080 × 1920. Like the source, its coded raster is 1920 × 1080 with a −90° display rotation. The source has 30 fps nominal and 99200/3307 fps average. The reconstructed movie contains no soundtrack.

## Rebuilding

Keep the original MOV and its matching `<ID>_vggt` directory side by side in `src/`. With Python 3, NumPy, OpenCV, Pillow, FFmpeg/ffprobe, a C compiler and Blender 4.2+ available, run the following from the delivered folder.

```sh
python3 src/reconstruct.py ~/Desktop/scene_reconstruction/src/img_1696.mov
```

The video path determines the reference directory and the output folder. The regular expression in `vg_reference.py` maps `img_<ID>.mov` to its sibling `<ID>_vggt/`. Camera and reference hashes are checked before using prepared geometry. The supplied authored model, traced textiles, scale anchors and artwork annotations belong to this physical scene. A different recording requires a new interpretation of its objects and scale. On another matched recording of the same archive format, the same command prepares fresh provenance, a source-specific camera track and a native unscaled reference-only Blender scene when Blender is available. These inspection inputs remain separate from the authored metric scene. The camera-preparation stage can also run on its own.

```sh
python3 src/prepare_camera.py ~/Desktop/scene_reconstruction/src/img_1696.mov
```

The archive includes the original prepared track and appearance assets, so regenerating them is optional. OpenCV is needed for camera feature tracking and room-finish preparation. `build.py` assembles the scene; `fit_img1696_alignment.py` reproduces the similarity; `vggt_audit.py` and `render_inspections.py` compute the mesh comparison and overview plates. `validate_clean.py`, `validate_dining_edges.py`, `compare_views.py` and `validate_view_motion.py` provide the readback, corner, visual and image-motion checks. `render_native.py` renders the geometry and `encode_render.py` restores the original presentation timing. On systems that cannot initialize graphics in a background Blender process, run the same native stages in a fresh graphical Blender session with a working graphics context.
