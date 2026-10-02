"""Dataset publication tests use fictional notes and temporary directories only."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_dataset


class DatasetPublicationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.vault = self.root / 'vault'
        self.output = self.root / 'data'
        self.output.mkdir()
        self.original = {
            'train.jsonl': b'old training data\n',
            'valid.jsonl': b'old validation data\n',
            'test.jsonl': b'old test data\n',
            'manifest.json': b'{"old_manifest": true}\n',
            'keep.txt': b'Unrelated user-owned file\n',
        }
        for name, content in self.original.items():
            (self.output / name).write_bytes(content)

    def assert_original(self):
        self.assertEqual({p.name: p.read_bytes() for p in self.output.iterdir()}, self.original)

    def populate(self):
        (self.vault / 'texts').mkdir(parents=True)
        book = self.vault / 'songs' / 'fictional'
        book.mkdir(parents=True)
        (self.vault / 'texts' / 'text--fixture.md').write_text(
            '---\ntext_key: fixture-text\n---\n# Fixture Text\n\n## Full Texts\n'
            'A fictional stanza for publication tests.\n\n## Appearances\nFixture book.\n', encoding='utf-8')
        (book / '1.md').write_text(
            '---\ntitle: Fixture Tune\nbook_family: fictional\nsong_no: 1\n'
            'text_key: fixture-text\n---\n# Fixture Tune\n- Raw First Line: A fictional stanza\n', encoding='utf-8')

    def run_main(self):
        with patch.object(sys, 'argv', ['prepare_dataset.py', '--vault-root', str(self.vault), '--output', str(self.output)]):
            with contextlib.redirect_stdout(io.StringIO()):
                prepare_dataset.main()

    def test_wrong_or_incorrectly_rooted_path_preserves_every_existing_output(self):
        for path in [self.vault, self.root]:
            with self.subTest(path=path):
                run = subprocess.run([sys.executable, str(Path(prepare_dataset.__file__)), '--vault-root', str(path), '--output', str(self.output)], capture_output=True, text=True)
                self.assertNotEqual(run.returncode, 0, run.stdout)
                self.assertIn('texts', run.stderr)
                self.assertIn('songs', run.stderr)
                self.assert_original()

    def test_empty_corpus_preserves_every_existing_output(self):
        (self.vault / 'texts').mkdir(parents=True)
        (self.vault / 'songs').mkdir()
        with self.assertRaises((ValueError, SystemExit)):
            with contextlib.redirect_stderr(io.StringIO()):
                self.run_main()
        self.assert_original()

    def test_blank_notes_do_not_make_an_empty_corpus_publishable(self):
        (self.vault / 'texts').mkdir(parents=True)
        book = self.vault / 'songs' / 'fictional'
        book.mkdir(parents=True)
        (book / 'blank.md').write_text(' \n\n', encoding='utf-8')
        with self.assertRaises((ValueError, SystemExit)):
            with contextlib.redirect_stderr(io.StringIO()):
                self.run_main()
        self.assert_original()

    def test_staging_write_failure_keeps_previous_dataset_intact(self):
        self.populate()
        original_open = Path.open
        def fail_validation(path, mode='r', *args, **kwargs):
            if path.name == 'valid.jsonl' and 'w' in mode:
                raise OSError('Simulated full destination filesystem')
            return original_open(path, mode, *args, **kwargs)
        with patch.object(Path, 'open', fail_validation):
            with self.assertRaisesRegex(OSError, 'full destination'):
                self.run_main()
        self.assert_original()
        self.assertFalse(list(self.root.glob('.data-stage-*')))

    def test_failed_promotion_rolls_back_every_file(self):
        self.populate()
        import os
        original_replace = os.replace
        def fail_promotion(source, destination):
            if Path(source).name == 'new':
                raise OSError('Simulated publication failure')
            return original_replace(source, destination)
        with patch.object(os, 'replace', fail_promotion):
            with self.assertRaisesRegex(OSError, 'publication failure'):
                self.run_main()
        self.assert_original()
        self.assertFalse(list(self.root.glob('.data-stage-*')))

    def test_managed_symlink_does_not_overwrite_its_external_target(self):
        self.populate()
        external = self.root / 'external-user-file.txt'
        external.write_bytes(b'External file remains unchanged\n')
        (self.output / 'train.jsonl').unlink()
        (self.output / 'train.jsonl').symlink_to(external)
        (self.output / 'keep-link').symlink_to(external)
        self.run_main()
        self.assertEqual(external.read_bytes(), b'External file remains unchanged\n')
        self.assertFalse((self.output / 'train.jsonl').is_symlink())
        self.assertTrue((self.output / 'keep-link').is_symlink())

    def test_failed_rollback_retains_backup_for_recovery(self):
        self.populate()
        import os
        original_replace = os.replace
        retained = []
        def fail_promotion_and_rollback(source, destination):
            if Path(source).name == 'new':
                raise OSError('Simulated promotion failure')
            if Path(source).name == 'previous':
                retained.append(Path(source))
                raise OSError('Simulated rollback failure')
            return original_replace(source, destination)
        with patch.object(os, 'replace', fail_promotion_and_rollback):
            with self.assertRaisesRegex(RuntimeError, 'previous data retained'):
                self.run_main()
        self.assertEqual(len(retained), 1)
        self.assertEqual({p.name: p.read_bytes() for p in retained[0].iterdir()}, self.original)
        self.assertFalse(self.output.exists())

    def test_symlink_output_is_rejected_without_touching_the_target(self):
        self.populate()
        previous = self.root / 'previous-dataset'
        self.output.rename(previous)
        self.output.symlink_to(previous, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'regular directory'):
            self.run_main()
        self.assert_original()

    def test_success_publishes_complete_consistent_dataset_and_preserves_other_files(self):
        self.populate()
        self.run_main()
        manifest = json.loads((self.output / 'manifest.json').read_text())
        splits = {name: [json.loads(line) for line in (self.output / f'{name}.jsonl').read_text().splitlines()] for name in ['train', 'valid', 'test']}
        self.assertEqual(manifest['total_examples'], 2)
        self.assertEqual(manifest['splits'], {name: len(records) for name, records in splits.items()})
        self.assertEqual(sum(map(len, splits.values())), 2)
        self.assertEqual(manifest['source_text_notes'], 1)
        self.assertEqual(manifest['source_song_notes'], 1)
        self.assertEqual((self.output / 'keep.txt').read_bytes(), self.original['keep.txt'])
        self.assertFalse(list(self.root.glob('.data-stage-*')))


if __name__ == '__main__':
    unittest.main()
