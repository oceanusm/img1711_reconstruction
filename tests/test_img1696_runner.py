"""Verify the second video's runner keeps inputs and experiments independent."""
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
SCRIPT = ROOT / "scripts/run_img1696_vggt.sh"
OVERRIDES = ("MAX_FRAMES", "IMAGE_RESOLUTION", "PREPROCESS_MODE", "VGGT_OUTPUT",
             "FRAMES_MANIFEST", "CHECKPOINT", "SCENE_RECON_BIN")


def runner_env(**overrides):
    env = {key: value for key, value in os.environ.items() if key not in OVERRIDES}
    env.update(overrides)
    return env


def option(argv, flag):
    return argv[argv.index(flag) + 1]


class Img1696RunnerTests(unittest.TestCase):
    def test_default_selection_spans_new_clip_and_preserves_point_budget(self):
        result = subprocess.run(["bash", str(SCRIPT), "--dry-run"],
                                env=runner_env(), capture_output=True, text=True, check=True)
        argv = shlex.split(result.stdout)
        for flag, value in (("--manifest", "data/img1696/frames/manifest.json"),
                            ("--stride", "1"), ("--max-frames", "450"),
                            ("--preprocess-mode", "balanced"), ("--image-resolution", "512"),
                            ("--precision", "bf16"), ("--point-stride", "1"),
                            ("--max-points", "5000000"), ("--confidence-percentile", "50"),
                            ("--output", "outputs/img1696/vggt_450_balanced512")):
            self.assertEqual(option(argv, flag), value)

    def test_changed_settings_have_distinct_output_paths(self):
        for values, expected in (({"MAX_FRAMES": "500"}, "vggt_500_balanced512"),
                                 ({"IMAGE_RESOLUTION": "768"}, "vggt_450_balanced768"),
                                 ({"PREPROCESS_MODE": "max_size"}, "vggt_450_max_size512")):
            with self.subTest(values=values):
                result = subprocess.run(["bash", str(SCRIPT), "--dry-run"],
                                        env=runner_env(**values), capture_output=True,
                                        text=True, check=True)
                self.assertEqual(option(shlex.split(result.stdout), "--output"),
                                 "outputs/img1696/" + expected)

    def test_dry_run_needs_no_assets_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            shutil.copy(SCRIPT, root / "scripts")
            before = sorted(root.rglob("*"))
            result = subprocess.run(["bash", str(root / "scripts" / SCRIPT.name), "--dry-run"],
                                    env=runner_env(), capture_output=True, text=True, check=True)
            self.assertIn("--max-frames 450", result.stdout)
            self.assertEqual(sorted(root.rglob("*")), before)

    def test_custom_paths_and_gpu_are_passed_unchanged(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            shutil.copy(SCRIPT, root / "scripts")
            assets = root / "input assets"
            assets.mkdir()
            manifest = assets / "frames manifest.json"
            manifest.write_text("{}")
            checkpoint = assets / "omega weights.pt"
            checkpoint.touch()
            executable = root / "mock scene-recon"
            executable.write_text(
                f"#!{sys.executable}\nimport json, os, sys\n"
                "with open(os.environ['MOCK_LOG'], 'w') as log:\n"
                "    json.dump({'args': sys.argv[1:], 'gpu': os.environ['CUDA_VISIBLE_DEVICES']}, log)\n")
            executable.chmod(0o755)
            log = root / "call.json"
            env = runner_env(MAX_FRAMES="500", IMAGE_RESOLUTION="768", PREPROCESS_MODE="balanced",
                             VGGT_OUTPUT="outputs/custom mapping", FRAMES_MANIFEST=str(manifest),
                             CHECKPOINT=str(checkpoint), SCENE_RECON_BIN=str(executable),
                             CUDA_VISIBLE_DEVICES="7", MOCK_LOG=str(log))
            subprocess.run(["bash", str(root / "scripts" / SCRIPT.name)],
                           env=env, capture_output=True, text=True, check=True)
            call = json.loads(log.read_text())
            self.assertEqual(call["gpu"], "7")
            for flag, value in (("--manifest", str(manifest)), ("--checkpoint", str(checkpoint)),
                                ("--output", "outputs/custom mapping"), ("--max-frames", "500"),
                                ("--image-resolution", "768")):
                self.assertEqual(option(call["args"], flag), value)

    def test_missing_manifest_fails_before_model_execution(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = str(Path(temporary) / "missing.json")
            result = subprocess.run(["bash", str(SCRIPT)],
                                    env=runner_env(FRAMES_MANIFEST=manifest),
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Missing frames manifest: " + manifest, result.stderr)

    def test_invalid_settings_fail_even_in_dry_run(self):
        for variable, values in (("MAX_FRAMES", ("0", "1", "-3", "5.5", "oops", "99999999999999999999")),
                                 ("IMAGE_RESOLUTION", ("0", "513", "-512", "512.0")),
                                 ("PREPROCESS_MODE", ("full",))):
            for value in values:
                with self.subTest(variable=variable, value=value):
                    result = subprocess.run(["bash", str(SCRIPT), "--dry-run"],
                                            env=runner_env(**{variable: value}),
                                            capture_output=True, text=True)
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(variable, result.stderr)
                    self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
