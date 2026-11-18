"""Verify experiment runners keep input selections and output paths consistent."""
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class GeometryRunnerTests(unittest.TestCase):
    def test_resolution_benchmark_uses_identical_view_selection_and_explicit_grids(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "not created"
            result = subprocess.run(
                ["bash", str(ROOT / "scripts/benchmark_vggt_resolution.sh"), "--dry-run", "--include-full"],
                env=dict(os.environ, FRAMES="24", OUTPUT_ROOT=str(output)),
                capture_output=True, text=True, check=True)
            commands = [shlex.split(line) for line in result.stdout.splitlines()]
            self.assertEqual(len(commands), 4)
            settings = []
            for argv in commands:
                option = lambda flag: argv[argv.index(flag) + 1]
                self.assertEqual(option("--stride"), "1")
                self.assertEqual(option("--max-frames"), "24")
                self.assertEqual(option("--manifest"), "data/frames/manifest.json")
                self.assertTrue(option("--output").startswith(str(output)))
                settings.append((option("--preprocess-mode"), option("--image-resolution")))
            self.assertEqual(settings, [("balanced", "512"), ("balanced", "768"),
                                        ("balanced", "1024"), ("max_size", "1920")])
            self.assertFalse(output.exists())

    def test_benchmark_rejects_accidental_unlimited_frame_selection(self):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts/benchmark_vggt_resolution.sh"), "--dry-run"],
            env=dict(os.environ, FRAMES="0"), capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not accept an unlimited", result.stderr)

    def test_depth_and_preview_follow_custom_output_grid_and_storage(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            shutil.copy(ROOT / "scripts/run_geometry.sh", root / "scripts")
            (root / "data/frames").mkdir(parents=True)
            (root / "data/frames/manifest.json").write_text("{}")
            executable = root / ".venv-depth/bin/scene-recon"
            executable.parent.mkdir(parents=True)
            executable.write_text(
                f"#!{sys.executable}\nimport json, os, sys\n"
                "with open(os.environ['MOCK_LOG'], 'a') as log:\n"
                "    log.write(json.dumps({'args': sys.argv[1:], 'gpu': os.environ['CUDA_VISIBLE_DEVICES']})+'\\n')\n")
            executable.chmod(0o755)
            log = root / "calls.jsonl"
            env = dict(os.environ, MOCK_LOG=str(log), CUDA_VISIBLE_DEVICES="7",
                       DEPTH_OUTPUT="outputs/high resolution", DEPTH_PROCESS_RES="1008",
                       DEPTH_CHUNK_SIZE="16", DEPTH_OVERLAP="4", DEPTH_COMPRESSION_LEVEL="0")
            env.pop("DEPTH_PREVIEW_OUTPUT", None)
            for stage in ("depth", "preview"):
                subprocess.run(["bash", str(root / "scripts/run_geometry.sh"), stage], env=env, check=True)
            depth, preview = [json.loads(line) for line in log.read_text().splitlines()]
            argv = depth["args"]
            for flag, value in (("--process-res", "1008"), ("--chunk-size", "16"),
                                ("--overlap", "4"), ("--compression-level", "0"),
                                ("--output", "outputs/high resolution")):
                self.assertEqual(argv[argv.index(flag) + 1], value)
            self.assertEqual(preview["args"], ["preview-depth", "--depth-index",
                             "outputs/high resolution/index.json", "--output",
                             "outputs/high resolution_preview.mp4"])
            self.assertEqual((depth["gpu"], preview["gpu"]), ("7", "7"))


if __name__ == "__main__":
    unittest.main()
