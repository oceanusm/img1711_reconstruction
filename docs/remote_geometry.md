# Run VGGT-Ω and Depth Anything 3 on the remote A6000

This workflow installs and runs only geometry/depth. SAM3 and Blender mask rendering can wait. Model APIs and CPU export contracts have been checked locally; successful A6000 inference is still to be verified.

## 1. Clone the code on the remote

Replace `YOUR_GITHUB_REPO_URL`, `USER@HOST`, and the remote directory below with your actual values. No Git remote has been created by this project.

```bash
git clone YOUR_GITHUB_REPO_URL ~/img1711_reconstruction
cd ~/img1711_reconstruction
```

If already cloned, run `git pull` inside that checkout instead.

## 2. Transfer the inputs from your Mac

**`data/`, weights, environments, and outputs are Git-ignored.** A Git push/pull transfers the code and configs, not the video or prepared images. Run this on the Mac after creating the remote checkout:

```bash
cd /Users/liumartin/Developer/img1711_reconstruction
rsync -av --progress data/ USER@HOST:~/img1711_reconstruction/data/
```

This transfers approximately 175 MB: the original video, 698 upright/tone-mapped JPEGs, their timestamp manifest, and the contact sheet. Keeping this exact manifest and frame set avoids differing FFmpeg versions changing later input/provenance comparisons. Preserve `data/frames/manifest.json` with the eventual model results and Blender scene.

Alternatively, copy only `data/IMG_1711.MOV`; the runner will extract frames on the remote if no manifest exists. That needs FFmpeg with `zscale` and `tonemap`. In that case use the newly generated remote manifest and frames as the common reference for all downstream work.

## 3. Install remote prerequisites and model environments

Target: Linux x86_64, Python **3.12** with venv support, Git, FFmpeg/ffprobe, and a working NVIDIA driver supporting the CUDA 12.6 PyTorch wheels. You do not need to compile/install the CUDA toolkit for this inference pipeline. Check the existing host first:

```bash
nvidia-smi
python3.12 --version
ffmpeg -version
```

On **Ubuntu 24.04**, missing OS packages can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg python3.12 python3.12-venv libgl1
```

Other distributions/releases may need a different Python installation. If you already have a Python 3.12 Conda environment, point the installer at its interpreter using `PYTHON_BIN=/absolute/path/to/python`. Do not replace a working GPU driver as part of these commands.

Minimal GPU container images may also need their distribution's GLib runtime for OpenCV. The installer's model-import checks will report missing system libraries before inference.

From the remote repository root:

```bash
bash scripts/setup_a6000.sh geometry
```

This creates `.venv-vggt` and `.venv-depth`, clones the pinned official model repositories, installs the CUDA PyTorch dependencies, tests imports and CUDA availability, and writes environment reports under `outputs/environment`. It does not install SAM3. Reserve tens of GB for dependencies, weights, and full-resolution numeric outputs.

## 4. Obtain weights

VGGT-Ω requires approved access to [facebook/VGGT-Omega](https://huggingface.co/facebook/VGGT-Omega). After access is granted, run on the remote:

```bash
.venv-vggt/bin/hf auth login
.venv-vggt/bin/hf download facebook/VGGT-Omega vggt_omega_1b_512.pt \
  --local-dir checkpoints/vggt-omega
```

DA3 downloads the public nested model automatically on its first run. The adapter pins the reviewed `DA3NESTED-GIANT-LARGE-1.1` checkpoint revision. You can run DA3 while waiting for VGGT access.

## 5. Run the models

Use a persistent terminal session such as `tmux` if disconnecting SSH would terminate your jobs. Run the stages sequentially on the same GPU.

All geometry/depth stages, including a viewable depth video:

```bash
bash scripts/run_geometry.sh
```

Or run individual stages:

```bash
# Full-video depth. This does not need the VGGT checkpoint.
bash scripts/run_geometry.sh depth
bash scripts/run_geometry.sh preview

# Joint reconstruction across the whole clip.
bash scripts/run_geometry.sh vggt
```

The corresponding model commands, useful for changing parameters, are:

```bash
.venv-vggt/bin/scene-recon vggt \
  --checkpoint checkpoints/vggt-omega/vggt_omega_1b_512.pt \
  --stride 10 --max-frames 72 --output outputs/vggt

.venv-depth/bin/scene-recon depth \
  --chunk-size 16 --overlap 4 --process-res 504 --output outputs/depth

.venv-depth/bin/scene-recon preview-depth
```

Defaults select **71 frames** across the full clip for one joint VGGT reconstruction and predict depth for **all 698 frames**. The 48 GB A6000 memory settings are starting values, not measured memory guarantees. To check the VGGT setup cheaply before the main run:

```bash
.venv-vggt/bin/scene-recon vggt \
  --checkpoint checkpoints/vggt-omega/vggt_omega_1b_512.pt \
  --max-frames 12 --output outputs/vggt_smoke
```

## Outputs to inspect and retain

| Path | Contents |
|---|---|
| `outputs/vggt/point_cloud.ply` | Colored points in one VGGT world coordinate system |
| `outputs/vggt/cameras.npz` | Timestamped camera poses and intrinsics; includes Blender camera-basis conversion |
| `outputs/vggt/index.json` | Source/frame mapping, scale and coordinate conventions, completion status |
| `outputs/vggt/depth/` | Per-selected-frame depth and confidence at model resolution |
| `outputs/depth/frames/` | Per-frame source-resolution float depth, confidence and cameras where available |
| `outputs/depth/index.json` | Depth units, frame mapping, and window coordinate scope |
| `outputs/depth/overlap_diagnostics.json` | Repeated-frame disagreements across neighboring windows |
| `outputs/depth_preview.mp4` | RGB beside depth, with one fixed color scale |
| `outputs/depth_preview.scale.json` | Preview color range and provenance |

Do not discard the raw float arrays after making the preview. Save the manifest, environment reports, and complete stage folders for comparison. To retrieve outputs, run from the Mac repository root:

```bash
rsync -av --progress USER@HOST:~/img1711_reconstruction/outputs/ outputs/
```

## Retrying and memory adjustments

The runner does **not** overwrite or silently skip previous outputs. If a later stage failed, rerun only that stage:

- DA3 resumes completed windows with the same arguments/input. Use `bash scripts/run_geometry.sh depth`, then `preview`.
- DA3 fingerprints frame sizes and modification times; transfer the inputs before starting and avoid replacing them during an interrupted run.
- VGGT is one joint inference; retry with a new `--output` if its directory contains partial files. It rejects nonempty output directories.
- Preview rejects existing MP4/scale files; give a new `--output` if you want another rendering.
- For OOM, reduce VGGT `--max-frames`; for DA3 use `--chunk-size 8 --overlap 2`, with a new output directory. Update preview `--depth-index` to that new directory's `index.json`.

The runner's `all` mode stops on the first error. If VGGT is waiting for access or needs troubleshooting, DA3 can run independently with the `depth` stage.

The next step after successful inference is [aligning and comparing the VGGT reconstruction with Blender](geometry_comparison.md). Cryptomatte and SAM masks are not required for the first camera/geometry comparison.

