"""Exercise the shell launcher without CUDA or a SAM model."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / 'scripts/run_dense_sam.sh'


class DenseSamLauncherTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.executable = self.root / 'fake python'
        self.log = self.root / 'invocation.json'
        self.executable.write_text(
            f'#!{sys.executable}\nimport json, os, sys\n'
            "with open(os.environ['DENSE_LAUNCHER_TEST_LOG'], 'w') as file:\n"
            "    json.dump({'argv': sys.argv[1:], 'visibility': os.environ.get('CUDA_VISIBLE_DEVICES'), "
            "'cwd': os.getcwd()}, file)\n")
        self.executable.chmod(0o755)

    def invoke(self, *args, **overrides):
        env = {key: value for key, value in os.environ.items()
               if key not in ('CUDA_VISIBLE_DEVICES', 'CUDA_VSIIBLE_DEVICES', 'PYTHON_BIN', 'SAM_CHECKPOINT')}
        env.update(PYTHON_BIN=str(self.executable), DENSE_LAUNCHER_TEST_LOG=str(self.log))
        env.update(overrides)
        result = subprocess.run(['bash', str(SCRIPT), *args], env=env,
                                capture_output=True, text=True, cwd=self.root)
        invocation = json.loads(self.log.read_text()) if self.log.exists() else None
        return result, invocation

    def test_default_uses_requested_three_cards_and_project_directory(self):
        result, invocation = self.invoke()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(invocation['argv'], ['-m', 'scene_reconstruction', 'sam-dense', '--gpus', '2,5,7'])
        self.assertEqual(invocation['cwd'], str(ROOT))
        self.assertIsNone(invocation['visibility'])

    def test_explicit_visibility_is_preserved_as_physical_gpu_list(self):
        for visible in ('7', '2,7'):
            with self.subTest(visible=visible):
                result, invocation = self.invoke('--resume', CUDA_VISIBLE_DEVICES=visible)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(invocation['visibility'], visible)
                self.assertEqual(invocation['argv'][-3:], ['--gpus', visible, '--resume'])

    def test_explicit_gpu_options_are_not_overridden(self):
        for arguments in (['--gpus', '2,7'], ['--gpus=2,7'],
                          ['--gpu-index', '1'], ['--gpu-index=1']):
            with self.subTest(arguments=arguments):
                result, invocation = self.invoke(*arguments)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(invocation['argv'][3:], arguments)

    def test_smoke_supplies_three_anchor_jobs_without_changing_gpu_selection(self):
        result, invocation = self.invoke('--smoke', '--resume')
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = invocation['argv']
        for flag, expected in (('--output', 'outputs/sam_dense_1711_parallel_smoke'),
                               ('--frame-limit', '8'), ('--anchor-count', '3'),
                               ('--grid-side', '8'), ('--max-per-anchor', '2'),
                               ('--track-batch-size', '2'), ('--gpus', '2,5,7')):
            self.assertEqual(argv[argv.index(flag)+1], expected)
        self.assertNotIn('--smoke', argv)
        self.assertIn('--resume', argv)

    def test_checkpoint_and_paths_with_spaces_are_passed_as_single_arguments(self):
        checkpoint = str(self.root / 'model weights.pt')
        result, invocation = self.invoke('--output', 'outputs/my masks', SAM_CHECKPOINT=checkpoint)
        self.assertEqual(result.returncode, 0, result.stderr)
        argv = invocation['argv']
        self.assertEqual(argv[argv.index('--output')+1], 'outputs/my masks')
        self.assertEqual(argv[argv.index('--checkpoint')+1], checkpoint)

    def test_empty_visibility_and_common_typo_fail_before_python(self):
        for env in ({'CUDA_VISIBLE_DEVICES': ''}, {'CUDA_VSIIBLE_DEVICES': '7'}):
            with self.subTest(env=env):
                result, invocation = self.invoke(**env)
                self.assertEqual(result.returncode, 2)
                self.assertIsNone(invocation)


if __name__ == '__main__':
    unittest.main()
