"""Exercise installer decisions with fake host tools; never install GPU packages."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
FAKE_TOOL = r'''
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import types

command = Path(sys.argv[0]).name
args = sys.argv[1:]
with open(os.environ["MOCK_LOG"], "a") as log:
    log.write(json.dumps({"command": command, "path": sys.argv[0], "args": args}) + "\n")
if command == "uname":
    print("Linux" if args == ["-s"] else "x86_64")
elif command in ("ffmpeg", "ffprobe", "nvidia-smi"):
    pass
elif command == "git":
    if args[0] == "clone":
        (Path(args[2]) / ".git").mkdir(parents=True)
    else:
        directory = Path(args[1])
        if args[2] == "checkout":
            (directory / ".git" / "revision").write_text(args[4])
        elif args[2] == "rev-parse":
            print((directory / ".git" / "revision").read_text())
        elif args[2] == "status":
            if os.environ.get("MOCK_GIT_DIRTY"):
                print(" M README.md")
        else:
            raise RuntimeError(args)
elif command == "scene-recon":
    assert args == ["doctor", "--gpu"]
    print('{"cuda_available": true}')
    sys.exit(int(os.environ.get("MOCK_DOCTOR_FAILURE", "0")))
elif command in ("python", "python3", "python3.12"):
    environment = Path(sys.argv[0]).parent.parent
    state_path = environment / "mock-packages.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    python_version = state.get("python", os.environ.get("MOCK_PYTHON_VERSION", "3.10"))
    if args[:2] == ["-m", "venv"]:
        directory = Path(args[2])
        (directory / "bin").mkdir(parents=True)
        for name in ("python", "scene-recon"):
            shutil.copyfile(__file__, directory / "bin" / name)
            (directory / "bin" / name).chmod(0o755)
        (directory / "mock-packages.json").write_text(json.dumps({"python": python_version}))
    elif args[:2] == ["-m", "pip"]:
        if args[2] == "install":
            for argument in args[3:]:
                if "==" in argument and argument.split("==")[0] in ("torch", "torchvision", "xformers"):
                    name, value = argument.split("==")
                    state[name] = value
            if "--index-url" in args:
                flavor = args[args.index("--index-url") + 1].rsplit("/", 1)[-1]
                state["cuda"] = {"cu124": "12.4", "cu126": "12.6"}[flavor]
            state_path.write_text(json.dumps(state))
        elif args[2] == "freeze":
            print("\n".join(f"{name}=={value}" for name, value in state.items()))
        else:
            assert args[2] == "check"
    elif args[0] == "-c":
        sys.version_info = tuple(map(int, python_version.split("."))) + (0, "final", 0)
        torch = types.ModuleType("torch")
        torch.__version__ = state.get("torch", "missing") + "+cu124"
        torch.version = types.SimpleNamespace(cuda=state.get("cuda"))
        vision = types.ModuleType("torchvision")
        vision.__version__ = state.get("torchvision", "missing")
        sys.modules.update(torch=torch, torchvision=vision)
        original_version = importlib.metadata.version
        importlib.metadata.version = lambda name: state[name] if name in state else original_version(name)
        for module_name, symbol in (("vggt_omega.models", "VGGTOmega"),
                                    ("depth_anything_3.api", "DepthAnything3"),
                                    ("sam3.model_builder", "build_sam3_video_predictor")):
            module = types.ModuleType(module_name)
            setattr(module, symbol, object)
            sys.modules[module_name] = module
        sys.argv = ["-c"] + args[2:]
        exec(args[1])
    else:
        raise RuntimeError(args)
else:
    raise RuntimeError(command)
'''


class SetupProfilesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="setup-profiles-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "scripts").mkdir()
        shutil.copyfile(ROOT / "scripts/setup_a6000.sh", self.root / "scripts/setup_a6000.sh")
        shutil.copytree(ROOT / "configs", self.root / "configs")
        self.tools = self.root / "fake-tools"
        self.tools.mkdir()
        for name in ("python3", "python3.12", "uname", "git", "ffmpeg", "ffprobe", "nvidia-smi"):
            path = self.tools / name
            path.write_text(f"#!{sys.executable}\n" + FAKE_TOOL)
            path.chmod(0o755)
        self.log = self.root / "commands.jsonl"
        self.environ = dict(os.environ, PATH=f"{self.tools}:{os.environ['PATH']}", MOCK_LOG=str(self.log))
        for name in ("CUDA_FLAVOR", "PYTHON_BIN", "MOCK_PYTHON_VERSION", "MOCK_DOCTOR_FAILURE", "MOCK_GIT_DIRTY"):
            self.environ.pop(name, None)

    def run_setup(self, *args, **environment):
        return subprocess.run(["bash", str(self.root / "scripts/setup_a6000.sh"), *args],
                              cwd=self.root, env=dict(self.environ, **environment),
                              capture_output=True, text=True)

    def calls(self):
        return [json.loads(line) for line in self.log.read_text().splitlines()] if self.log.exists() else []

    def installs(self):
        return [call for call in self.calls() if call["args"][:3] == ["-m", "pip", "install"]]

    def test_default_geometry_installs_matching_cu124_wheels_and_reuses(self):
        result = self.run_setup()
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("vggt", "depth"):
            state = json.loads((self.root / f".venv-{name}/mock-packages.json").read_text())
            self.assertEqual((state["python"], state["torch"], state["torchvision"], state["cuda"]),
                             ("3.10", "2.6.0", "0.21.0", "12.4"))
            profile = (self.root / f"outputs/environment/{name}-profile.txt").read_text()
            self.assertIn("cuda=cu124\n", profile)
            self.assertEqual(profile, (self.root / f".venv-{name}/.scene-recon-profile").read_text())
        self.assertEqual(state["xformers"], "0.0.29.post3")
        self.assertFalse((self.root / ".venv-sam").exists())
        wheel_installs = [c for c in self.installs() if "--index-url" in c["args"]]
        self.assertEqual(len(wheel_installs), 3)
        self.assertTrue(all("https://download.pytorch.org/whl/cu124" in c["args"] for c in wheel_installs))
        self.log.write_text("")
        result = self.run_setup("geometry")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.installs(), [])
        self.assertFalse(any(c["args"][:2] == ["-m", "venv"] for c in self.calls()))
        self.assertIn("Reusing depth", result.stdout)

    def test_sam_and_all_cu124_fail_before_host_commands_or_mutations(self):
        for target in ("sam", "all"):
            with self.subTest(target=target):
                result = self.run_setup(target, "--recreate-envs")
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("explicit cu126", result.stderr)
                self.assertEqual(self.calls(), [])
                self.assertFalse((self.root / "vendor").exists())

    def test_explicit_cu126_sam_requires_python312(self):
        result = self.run_setup("sam", CUDA_FLAVOR="cu126")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SAM3 requires Python 3.12", result.stderr)
        self.assertFalse((self.root / "vendor").exists())
        result = self.run_setup("sam", CUDA_FLAVOR="cu126", PYTHON_BIN="python3.12", MOCK_PYTHON_VERSION="3.12")
        self.assertEqual(result.returncode, 0, result.stderr)
        state = json.loads((self.root / ".venv-sam/mock-packages.json").read_text())
        self.assertEqual((state["torch"], state["torchvision"], state["cuda"]), ("2.7.1", "0.22.1", "12.6"))

    def test_legacy_second_environment_blocks_all_installs(self):
        legacy = self.root / ".venv-depth"
        legacy.mkdir()
        (legacy / "keep-me").write_text("previous installation")
        result = self.run_setup("geometry")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--recreate-envs", result.stderr)
        self.assertEqual((legacy / "keep-me").read_text(), "previous installation")
        self.assertFalse((self.root / ".venv-vggt").exists())
        self.assertFalse((self.root / "vendor").exists())
        self.assertEqual(self.installs(), [])

    def test_recreate_archives_only_selected_envs_without_overwriting_backups(self):
        for name in ("vggt", "depth", "sam"):
            directory = self.root / f".venv-{name}"
            directory.mkdir()
            (directory / "keep-me").write_text(name)
        result = self.run_setup("geometry", "--recreate-envs")
        self.assertEqual(result.returncode, 0, result.stderr)
        for name in ("vggt", "depth"):
            backups = list(self.root.glob(f".venv-{name}.backup-*/environment/keep-me"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(backups[0].read_text(), name)
        self.assertEqual((self.root / ".venv-sam/keep-me").read_text(), "sam")
        result = self.run_setup("vggt", "--recreate-envs")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(list(self.root.glob(".venv-vggt.backup-*"))), 2)
        self.assertEqual(len(list(self.root.glob(".venv-depth.backup-*"))), 1)
        self.assertEqual(len(list(self.root.glob(".venv-sam.backup-*"))), 0)

    def test_profile_switch_and_changed_runtime_both_require_recreation(self):
        result = self.run_setup("vggt")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.log.write_text("")
        result = self.run_setup("vggt", CUDA_FLAVOR="cu126")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--recreate-envs", result.stderr)
        self.assertEqual(self.installs(), [])
        state_path = self.root / ".venv-vggt/mock-packages.json"
        state = json.loads(state_path.read_text())
        state["cuda"] = "12.6"
        state_path.write_text(json.dumps(state))
        result = self.run_setup("vggt")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("CUDA build differs", result.stderr)
        self.assertEqual(self.installs(), [])
        self.assertEqual(json.loads(state_path.read_text())["cuda"], "12.6")

    def test_failed_gpu_check_never_marks_partial_environment_complete(self):
        result = self.run_setup("vggt", MOCK_DOCTOR_FAILURE="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / ".venv-vggt/.scene-recon-profile").exists())
        self.assertFalse((self.root / "outputs/environment/vggt-profile.txt").exists())
        self.log.write_text("")
        result = self.run_setup("vggt")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("--recreate-envs", result.stderr)
        self.assertEqual(self.installs(), [])

    def test_symlink_environment_is_never_followed_or_archived(self):
        outside = self.root / "other-environment"
        outside.mkdir()
        (outside / "keep-me").write_text("untouched")
        (self.root / ".venv-vggt").symlink_to(outside, target_is_directory=True)
        result = self.run_setup("vggt", "--recreate-envs")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("symlink", result.stderr)
        self.assertEqual((outside / "keep-me").read_text(), "untouched")
        self.assertFalse((self.root / "vendor").exists())

    def test_dirty_source_blocks_archiving(self):
        result = self.run_setup("vggt")
        self.assertEqual(result.returncode, 0, result.stderr)
        (self.root / ".venv-vggt/keep-me").write_text("untouched")
        self.log.write_text("")
        result = self.run_setup("vggt", "--recreate-envs", MOCK_GIT_DIRTY="1")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Uncommitted upstream edits", result.stderr)
        self.assertEqual((self.root / ".venv-vggt/keep-me").read_text(), "untouched")
        self.assertEqual(list(self.root.glob(".venv-vggt.backup-*")), [])
        self.assertEqual(self.installs(), [])


if __name__ == "__main__":
    unittest.main()
