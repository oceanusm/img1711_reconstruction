# Original scene baselines

- `before_vqa/scene.blend`: generated without advance knowledge of the VQA task.
- `vqa_informed/scene.blend`: generated with advance knowledge of VQA and its questions.
- `vggt_guided_1711/scene.blend`: the model was instructed to fit the Blender reconstruction roughly to the VGGT point cloud. Agreement with that cloud measures adherence to the construction reference, not independent reconstruction accuracy. Its viewer description is recorded in `baselines.json` and matched by the source scene's SHA256 when generating viewers.

These are the clean inputs supplied by the user. The comparison/export tools read them and write derived artifacts under `outputs/`; they never save changes into these files. `baselines.json` records their initial SHA256 hashes. Keep experimental scene edits as separately named copies so comparisons against the originals remain reproducible.

The baseline condition comes from the user's description, not an inference from the scene geometry. Differences in model scores are comparisons against imperfect reconstruction/depth estimates, not certified ground-truth errors.
