"""Preset overrides, safe reruns, and explicit physical-GPU selection."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scene_reconstruction.cli import parser
from scene_reconstruction.vggt_omega import (
    apply_preset, check_output_directory, select_frames, validate_gpu_environment,
)


class VggtProfileTests(unittest.TestCase):
    def options(self, *extra):
        return apply_preset(parser().parse_args(["vggt", "--checkpoint", "model.pt", *extra]))

    def test_quality_and_quick_preserve_full_clip_endpoints(self):
        frames = [{"index": i} for i in range(698)]
        for preset, expected_count in (("quality", 350), ("quick", 71)):
            with self.subTest(preset=preset):
                args = self.options("--preset", preset)
                selected = select_frames(frames, args.stride, args.max_frames)
                self.assertEqual(len(selected), expected_count)
                self.assertEqual((selected[0]["index"], selected[-1]["index"]), (0, 697))
        self.assertEqual(self.options().preprocess_mode, "balanced")

    def test_explicit_zero_cap_and_stride_one_retain_every_frame(self):
        for preset in ("quality", "quick"):
            args = self.options("--preset", preset, "--max-frames", "0", "--stride", "1",
                                "--preprocess-mode", "max_size", "--max-points", "123")
            frames = [{"index": i} for i in range(698)]
            self.assertEqual(select_frames(frames, args.stride, args.max_frames), frames)
            self.assertEqual(args.preprocess_mode, "max_size")
            self.assertEqual(args.max_points, 123)

    def test_complete_and_partial_outputs_are_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            check_output_directory(root)
            marker = root / "cameras.npz"
            marker.write_bytes(b"retain original export")
            with self.assertRaisesRegex(FileExistsError, "partial results"):
                check_output_directory(root)
            (root / "index.json").write_text(json.dumps({"status": "complete"}))
            with self.assertRaisesRegex(FileExistsError, "completed reconstruction"):
                check_output_directory(root)
            self.assertEqual(marker.read_bytes(), b"retain original export")
            self.assertEqual(json.loads((root / "index.json").read_text())["status"], "complete")

    def test_misspelled_gpu_variable_fails_instead_of_silently_using_other_card(self):
        with patch.dict(os.environ, {"CUDA_VSIIBLE_DEVICES": "7"}, clear=True):
            with self.assertRaisesRegex(ValueError, "CUDA_VISIBLE_DEVICES=7"):
                validate_gpu_environment()
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "7"}, clear=True):
            validate_gpu_environment()

    def test_wrapper_forwards_preset_output_and_gpu_without_splitting_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "scripts").mkdir()
            shutil.copy(Path(__file__).resolve().parents[1] / "scripts/run_geometry.sh", root / "scripts")
            (root / "data/frames").mkdir(parents=True)
            (root / "data/frames/manifest.json").write_text("{}")
            executable = root / ".venv-vggt/bin/scene-recon"
            executable.parent.mkdir(parents=True)
            executable.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$MOCK_ARGS"\nprintf "%s" "$CUDA_VISIBLE_DEVICES" > "$MOCK_GPU"\n')
            executable.chmod(0o755)
            args_log, gpu_log = root / "args", root / "gpu"
            env = dict(os.environ, CUDA_VISIBLE_DEVICES="7", VGGT_PRESET="quick",
                       VGGT_OUTPUT="outputs/separate quality run", MOCK_ARGS=str(args_log), MOCK_GPU=str(gpu_log))
            subprocess.run(["bash", str(root / "scripts/run_geometry.sh"), "vggt"], env=env, check=True)
            argv = args_log.read_text().splitlines()
            self.assertEqual(argv[argv.index("--preset") + 1], "quick")
            self.assertEqual(argv[argv.index("--output") + 1], "outputs/separate quality run")
            self.assertEqual(gpu_log.read_text(), "7")


if __name__ == "__main__":
    unittest.main()
