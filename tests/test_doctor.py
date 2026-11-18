import contextlib
import io
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scene_reconstruction.doctor import command_report, run


class DoctorTests(unittest.TestCase):
    def test_missing_optional_command_is_reported(self):
        with patch("scene_reconstruction.doctor.shutil.which", return_value=None):
            self.assertEqual(command_report(["nvcc", "--version"]), {"available": False})

    def test_command_failure_does_not_hide_diagnostics(self):
        result = types.SimpleNamespace(returncode=1, stdout="", stderr="driver unavailable\n")
        with patch("scene_reconstruction.doctor.shutil.which", return_value="/usr/bin/nvidia-smi"), \
             patch("scene_reconstruction.doctor.subprocess.run", return_value=result):
            report = command_report(["nvidia-smi"])
        self.assertEqual(report["returncode"], 1)
        self.assertEqual(report["error"], "driver unavailable")

    def test_cpu_report_needs_no_gpu_stack(self):
        with tempfile.TemporaryDirectory() as directory:
            capture = io.StringIO()
            with contextlib.redirect_stdout(capture):
                run(types.SimpleNamespace(manifest=Path(directory) / "absent.json", gpu=False))
            report = json.loads(capture.getvalue())
        self.assertEqual(report["frames"], "Not extracted yet")
        self.assertNotIn("gpu_smoke_check", report)


if __name__ == "__main__":
    unittest.main()
