"""Deterministic grouped split policy checks; no corpus or training required."""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import prepare_dataset


class SplitPolicyTests(unittest.TestCase):
    def test_every_exact_integer_boundary(self):
        space = 1 << 160
        valid_start = (8 * space + 9) // 10
        test_start = (9 * space + 9) // 10
        for value, expected in [(0, 'train'), (valid_start - 1, 'train'), (valid_start, 'valid'), (test_start - 1, 'valid'), (test_start, 'test'), (space - 1, 'test')]:
            with self.subTest(value=value, expected=expected):
                digest = value.to_bytes(20, 'big')
                with patch.object(prepare_dataset.hashlib, 'sha1', return_value=SimpleNamespace(digest=lambda: digest)):
                    split = prepare_dataset.split_examples([{'group': 'controlled-boundary', 'id': 'fixture'}])
                self.assertEqual([name for name, records in split.items() if records], [expected])

    def test_groups_stay_together_and_do_not_depend_on_input_order(self):
        examples = [{'group': f'fixture-group-{group}', 'id': f'{group}-{member}'} for group in range(1000) for member in range(3)]
        def assignments(records):
            return {example['id']: name for name, entries in prepare_dataset.split_examples(records).items() for example in entries}
        original = assignments(examples)
        self.assertEqual(original, assignments(list(reversed(examples))))
        for group in range(1000):
            self.assertEqual(len({original[f'{group}-{member}'] for member in range(3)}), 1)
        self.assertEqual(set(original.values()), {'train', 'valid', 'test'})

    def test_large_deterministic_group_fixture_matches_documented_ratios(self):
        count = 100_000
        splits = prepare_dataset.split_examples([{'group': f'ratio-fixture-{index}', 'id': index} for index in range(count)])
        for name, target in [('train', 0.8), ('valid', 0.1), ('test', 0.1)]:
            self.assertAlmostEqual(len(splits[name]) / count, target, delta=0.005)

    def test_manifest_records_the_policy_version_and_thresholds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'vault' / 'texts').mkdir(parents=True)
            (root / 'vault' / 'songs').mkdir()
            (root / 'vault' / 'texts' / 'fixture.md').write_text('# Fixture\n## Full Texts\nFictional text.\n## Appearances\nFixture.\n')
            with patch.object(sys, 'argv', ['prepare_dataset.py', '--vault-root', str(root / 'vault'), '--output', str(root / 'data')]):
                with contextlib.redirect_stdout(io.StringIO()):
                    prepare_dataset.main()
            manifest = json.loads((root / 'data' / 'manifest.json').read_text())
            self.assertEqual(manifest['split_policy']['version'], 'sha1-160-80-10-10-v2')
            self.assertEqual(manifest['split_policy']['fractions'], {'train': 0.8, 'valid': 0.1, 'test': 0.1})
            self.assertEqual(manifest['split_policy']['grouped'], True)


if __name__ == '__main__':
    unittest.main()
