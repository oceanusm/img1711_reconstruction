# Run VGGT-Ω and Depth Anything 3 on the remote A6000

This workflow installs and runs only geometry/depth. SAM3 and Blender mask rendering can wait. Model APIs and CPU export contracts have been checked locally; successful A6000 inference is still to be verified.

## 1. Clone the code on the remote

Replace `YOUR_GITHUB_REPO_URL`, `USER@HOST`, and the remote directory below with your actual values.

```bash
git clone YOUR_GITHUB_REPO_URL ~/img1711_reconstruction
cd ~/img1711_reconstruction
```

If already cloned, run `git pull` inside that checkout instead.

## 2. Verify or transfer the inputs

The video, canonical frames, manifest and both Blender originals are now tracked in this checkout. After pulling the commit containing them, verify `data/frames/manifest.json` and `data/frames/000697.jpg` exist. **Weights, environments and generated outputs remain Git-ignored.** If the remote lacks the prepared inputs, transfer them from the Mac:

```bash
cd /Users/liumartin/Developer/img1711_reconstruction
rsync -av --progress data/ USER@HOST:~/img1711_reconstruction/data/
```

This transfers approximately 175 MB: the original video, 698 upright/tone-mapped JPEGs, their timestamp manifest, and the contact sheet. Keeping this exact manifest and frame set avoids differing FFmpeg versions changing later input/provenance comparisons. Preserve `data/frames/manifest.json` with the eventual model results and Blender scene.

Alternatively, copy only `data/IMG_1711.MOV`; the runner will extract frames on the remote if no manifest exists. That needs FFmpeg with `zscale` and `tonemap`. In that case use the newly generated remote manifest and frames as the common reference for all downstream work.

## 3. Install remote prerequisites and model environments

Target: **Ubuntu 22.04, Linux x86_64, CUDA 12.4**, Python **3.10–3.12** with venv support, Git and FFmpeg/ffprobe. The reported remote has driver **550.54.14** and Python **3.12.11**, which match this geometry profile. You do not need to replace Python or compile/install the CUDA toolkit. Check the host first:

```bash
nvidia-smi
python3 --version
ffmpeg -version
```

On **Ubuntu 22.04**, missing OS packages can be installed with:

```bash
sudo apt-get update
sudo apt-get install -y git ffmpeg python3-venv libgl1 libglib2.0-0
```

Ubuntu 22.04's system Python 3.10 is supported for geometry. The already active Conda Python 3.12.11 is also supported: `PYTHON_BIN=python3` uses the interpreter on your PATH. A different interpreter can be selected with `PYTHON_BIN=/absolute/path/to/python`. SAM3's separate newer environment is outside this first geometry run.

The three CUDA version readings mean different things: `nvidia-smi` shows driver support, `nvcc --version` shows an optional installed toolkit, and `torch.version.cuda` identifies the PyTorch runtime build. The earlier installer selected CUDA 12.6 wheels inside the project venvs; it did not change the host's CUDA toolkit or driver. CUDA minor-version compatibility can allow those wheels to run on some older 12.x drivers, with limitations; their presence alone is not proof of a failed installation. This updated profile deliberately uses CUDA 12.4 wheels. [NVIDIA compatibility rules](https://docs.nvidia.com/deploy/cuda-compatibility/latest/minor-version-compatibility.html)

The supplied GPU snapshot showed GPU 7 essentially empty, while other GPUs were occupied. If GPU 7 is still available for this job, select it before setup checks and inference:

```bash
export CUDA_VISIBLE_DEVICES=7
```

PyTorch then calls that physical card logical `cuda:0`. Do not additionally pass `--device cuda:7` after masking to a single GPU.

From the remote repository root:

```bash
CUDA_FLAVOR=cu124 PYTHON_BIN=python3 bash scripts/setup_a6000.sh geometry
```

This creates `.venv-vggt` and `.venv-depth`, clones the pinned official sources, and uses **torch 2.6.0 / torchvision 0.21.0 / xformers 0.0.29.post3**, with CUDA 12.4 wheels. It tests imports and small CUDA matrix/attention kernels, then records environment reports under `outputs/environment`. It does not install SAM3. Reserve tens of GB for dependencies, weights, and full-resolution numeric outputs. [Official PyTorch CUDA 12.4 installation](https://pytorch.org/get-started/previous-versions/#v260)

### If the previous installer already created environments

After pushing these changes and pulling them on the remote, recreate the project venvs using:

```bash
cd /local1/oceanusm/img1711_reconstruction
git pull
export CUDA_VISIBLE_DEVICES=7
CUDA_FLAVOR=cu124 PYTHON_BIN=python3 \
  bash scripts/setup_a6000.sh geometry --recreate-envs
```

The flag moves existing `.venv-vggt` and `.venv-depth` into uniquely named backup directories before constructing fresh environments. It does not delete them or alter the active Conda environment, data, model weights, or results. Run this when no job is using those project venvs. Without the flag, an old/unrecorded or incompatible venv is rejected rather than silently mixing CUDA/PyTorch packages. A later ordinary rerun can reuse matching environments.

Inspect the installed runtimes and selected GPU:

```bash
.venv-vggt/bin/scene-recon doctor --gpu
.venv-depth/bin/scene-recon doctor --gpu
```

Expect torch 2.6.0 (possibly with a `+cu124` suffix), `torch_cuda: "12.4"`, and successful small kernel checks. These checks do not establish model accuracy or full inference memory requirements. To deliberately retain the older package profile on an appropriately supported host, select `CUDA_FLAVOR=cu126`; SAM/all installation additionally requires Python 3.12.

## 4. Obtain weights

Download VGGT-Ω from the public [wincentIsMe/VGGT-Omega mirror](https://huggingface.co/wincentIsMe/VGGT-Omega), whose model card describes an unmodified copy of the selected **VGGT-Omega-1B-512** (`vggt_omega_1b_512.pt`) checkpoint. It does not require the official repository's per-account access approval. Model code still comes from the pinned official GitHub source. The 512 checkpoint is the **in-the-wild** model; the separate `VGGT-Omega-1B-416-Reproduction` checkpoint is for benchmarking and is not used by these commands. [Official model table](https://github.com/facebookresearch/vggt-omega#pretrained-models)

Run on the remote:

```bash
.venv-vggt/bin/hf download wincentIsMe/VGGT-Omega vggt_omega_1b_512.pt \
  --local-dir checkpoints/vggt-omega
```

DA3 downloads the public nested model automatically on its first run. The adapter pins the reviewed `DA3NESTED-GIANT-LARGE-1.1` checkpoint revision. DA3 can run independently of the VGGT checkpoint download.

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

The runner's `all` mode stops on the first error. If VGGT is still downloading or needs troubleshooting, DA3 can run independently with the `depth` stage.

The next step after successful inference is [aligning and comparing the VGGT reconstruction with Blender](geometry_comparison.md). Cryptomatte and SAM masks are not required for the first camera/geometry comparison.

