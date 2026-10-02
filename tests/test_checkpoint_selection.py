"""Checkpoint promotion fixtures contain arbitrary bytes, not trained models."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'openclaw_capability' / 'select_best_checkpoint.py'
spec = importlib.util.spec_from_file_location('checkpoint_selection', SCRIPT)
selection = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selection)


class CheckpointSelectionTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.adapter = self.root / 'adapter'
        self.adapter.mkdir()
        self.old = b'Previously usable canonical fixture\n'
        self.canonical = self.adapter / 'adapters.safetensors'
        self.canonical.write_bytes(self.old)
        self.checkpoint = self.adapter / '0000040_adapters.safetensors'
        self.new = b'Numbered selected checkpoint fixture\n' * 128
        self.checkpoint.write_bytes(self.new)
        (self.adapter / '0000020_adapters.safetensors').write_bytes(b'Other checkpoint')
        self.log = self.root / 'training.log'
        self.log.write_text('Iter 1: Val loss 0.01\nIter 20: Val loss 0.9\nIter 40: Val loss 0.4\n')
        self.receipt = self.root / 'reports' / 'checkpoint_selection.json'

    def run_main(self):
        with patch.object(selection, 'HERE', self.root), patch.object(sys, 'argv', ['select_best_checkpoint.py', '--log', str(self.log), '--adapter', str(self.adapter)]):
            with contextlib.redirect_stdout(io.StringIO()):
                return selection.main()

    def assert_no_staging(self):
        self.assertEqual(list(self.adapter.glob('.*.tmp')), [])
        self.assertEqual(list(self.receipt.parent.glob('.*.tmp')), [])

    def test_interrupted_copy_preserves_prior_canonical_and_source(self):
        copied = []
        def interrupted(source, destination):
            copied.append(Path(destination))
            Path(destination).write_bytes(b'partial')
            raise OSError('Interrupted copy fixture')
        with patch.object(selection.shutil, 'copy2', interrupted):
            with self.assertRaisesRegex(OSError, 'Interrupted copy'):
                self.run_main()
        self.assertEqual(self.canonical.read_bytes(), self.old)
        self.assertEqual(self.checkpoint.read_bytes(), self.new)
        self.assertEqual(copied[0].parent, self.adapter)
        self.assertNotEqual(copied[0], self.canonical)
        self.assert_no_staging()

    def test_corrupt_copy_is_rejected_before_replacing_canonical(self):
        self.receipt.parent.mkdir()
        def corrupt(source, destination):
            Path(destination).write_bytes(b'X' * len(self.new))
            return destination
        with patch.object(selection.shutil, 'copy2', corrupt):
            with self.assertRaisesRegex(RuntimeError, 'verification'):
                self.run_main()
        self.assertEqual(self.canonical.read_bytes(), self.old)
        self.assert_no_staging()

    def test_empty_checkpoint_is_rejected_without_replacing_canonical(self):
        self.receipt.parent.mkdir()
        self.checkpoint.write_bytes(b'')
        with self.assertRaisesRegex(ValueError, 'empty'):
            self.run_main()
        self.assertEqual(self.canonical.read_bytes(), self.old)
        self.assert_no_staging()

    def test_receipt_staging_failure_preserves_canonical(self):
        with patch.object(selection.json, 'dumps', side_effect=OSError('Receipt staging failed')):
            with self.assertRaisesRegex(OSError, 'Receipt staging'):
                self.run_main()
        self.assertEqual(self.canonical.read_bytes(), self.old)
        self.assert_no_staging()

    def test_atomic_adapter_replace_failure_preserves_old_adapter_and_receipt(self):
        self.receipt.parent.mkdir()
        self.receipt.write_bytes(b'Prior receipt\n')
        replace = os.replace
        def fail_adapter(source, destination):
            if Path(destination) == self.canonical:
                raise OSError('Adapter replace failed')
            return replace(source, destination)
        with patch.object(os, 'replace', fail_adapter):
            with self.assertRaisesRegex(OSError, 'Adapter replace'):
                self.run_main()
        self.assertEqual(self.canonical.read_bytes(), self.old)
        self.assertEqual(self.receipt.read_bytes(), b'Prior receipt\n')
        self.assert_no_staging()

    def test_creates_report_directory_and_publishes_matching_verified_receipt(self):
        self.assertEqual(self.run_main(), 0)
        self.assertEqual(self.canonical.read_bytes(), self.new)
        self.assertEqual(self.checkpoint.read_bytes(), self.new)
        receipt = json.loads(self.receipt.read_text())
        self.assertEqual(receipt['best_iteration'], 40)
        self.assertEqual(receipt['best_validation_loss'], 0.4)
        self.assertEqual(receipt['adapter_sha256'], hashlib.sha256(self.new).hexdigest())
        self.assertEqual(receipt['adapter_bytes'], len(self.new))
        self.assertEqual(len(receipt['all_validation_losses']), 3)
        self.assert_no_staging()

    def test_receipt_replace_failure_retains_prior_receipt_and_complete_new_adapter(self):
        self.receipt.parent.mkdir()
        self.receipt.write_bytes(b'Prior receipt\n')
        replace = os.replace
        def fail_receipt(source, destination):
            if Path(destination) == self.receipt:
                raise OSError('Receipt replace failed')
            return replace(source, destination)
        with patch.object(os, 'replace', fail_receipt):
            with self.assertRaisesRegex(OSError, 'Receipt replace'):
                self.run_main()
        self.assertEqual(self.canonical.read_bytes(), self.new)
        self.assertEqual(self.receipt.read_bytes(), b'Prior receipt\n')
        self.assert_no_staging()


if __name__ == '__main__':
    unittest.main()
