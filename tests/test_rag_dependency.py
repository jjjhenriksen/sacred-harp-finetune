"""Validate startup/retrieval using an external fixture module, without MLX."""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from provider_fixture import load_provider

provider = load_provider()
FIXTURE = '''
build_count = 0
def build_collection(refresh):
    global build_count
    build_count += 1
    return {'fixture': True}
def exact_verse_answer(query):
    return None
def retrieve(collection, query):
    assert collection == {'fixture': True}
    return [{'source': 'fixture.txt', 'text': 'Fictional verse', 'distance': 0.125}]
def structured_song_indexes():
    return {}, {}
def _metadata_fields(text):
    return {}
def _canonical_sections(text):
    return {}
def _section_for_witness(sections, book, number, source):
    return None
'''


class RagDependencyTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.model = self.root / 'model'
        self.adapter = self.root / 'adapter'
        self.model.mkdir()
        self.adapter.mkdir()
        self.backend = self.root / 'fixture_rag.py'
        self.backend.write_text(FIXTURE)
        self.native = Mock(return_value=(object(), Mock()))
        patcher = patch.object(provider, 'load', self.native)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.previous_modules = set(sys.modules)
        self.addCleanup(self.clean_fixture_modules)

    def clean_fixture_modules(self):
        for name in set(sys.modules) - self.previous_modules:
            if name.startswith('sacred_harp_openclaw_rag_'):
                del sys.modules[name]

    def runtime(self):
        return provider.SpecialistRuntime(self.model, self.adapter, rag_script=self.backend)

    def assert_startup_error(self, message):
        with self.assertRaisesRegex(RuntimeError, message) as failure:
            self.runtime()
        self.assertIn('--rag-script', str(failure.exception))
        self.assertIn(str(self.backend), str(failure.exception))
        self.native.assert_not_called()
        self.assertFalse(any(name.startswith('sacred_harp_openclaw_rag_') for name in set(sys.modules) - self.previous_modules))

    def test_missing_default_backend_fails_before_model_load(self):
        self.backend.unlink()
        with patch.object(provider, 'RAG_SCRIPT', self.backend):
            with self.assertRaisesRegex(RuntimeError, '--rag-script'):
                provider.SpecialistRuntime(self.model, self.adapter)
        self.native.assert_not_called()

    def test_missing_dependency_is_actionable_before_model_load(self):
        self.backend.unlink()
        self.assert_startup_error('not found')

    def test_missing_or_noncallable_interface_is_rejected(self):
        for name in ['build_collection', 'exact_verse_answer', 'retrieve', 'structured_song_indexes', '_metadata_fields', '_canonical_sections', '_section_for_witness']:
            with self.subTest(name=name):
                self.backend.write_text(FIXTURE + f'\n{name} = None\n')
                self.assert_startup_error(name)

    def test_import_failure_retains_cause_and_cleans_module(self):
        self.backend.write_text('import definitely_absent_sacred_harp_fixture_dependency\n')
        with self.assertRaises(RuntimeError) as failure:
            self.runtime()
        self.assertIsInstance(failure.exception.__cause__, ModuleNotFoundError)
        self.assertIn('--rag-script', str(failure.exception))
        self.native.assert_not_called()
        self.assertFalse(any(name.startswith('sacred_harp_openclaw_rag_') for name in set(sys.modules) - self.previous_modules))

    def test_failed_collection_and_invalid_indexes_fail_startup(self):
        for extra, message in [
            ('def build_collection(refresh):\n    raise RuntimeError("fixture corpus unavailable")\n', 'fixture corpus unavailable'),
            ('def build_collection(refresh):\n    return None\n', 'collection'),
            ('def structured_song_indexes():\n    return [], {}\n', 'indexes')]:
            with self.subTest(message=message):
                self.backend.write_text(FIXTURE + '\n' + extra)
                self.assert_startup_error(message)

    def test_injected_backend_is_ready_and_reused_for_search(self):
        runtime = self.runtime()
        self.assertEqual(runtime._rag.build_count, 1)
        self.native.assert_called_once_with(str(self.model), adapter_path=str(self.adapter))
        expected = {'query': 'Fictional query', 'exact': False, 'results': [{'source': 'fixture.txt', 'text': 'Fictional verse', 'distance': 0.125}]}
        self.assertEqual(runtime.search('Fictional query'), expected)
        self.assertEqual(runtime.search('Fictional query'), expected)
        self.assertEqual(runtime._rag.build_count, 1)

    def test_cli_backend_failure_never_binds_or_advertises_readiness(self):
        self.backend.unlink()
        argv = ['provider', '--model', str(self.model), '--adapter', str(self.adapter), '--rag-script', str(self.backend), '--port', '0']
        output = io.StringIO()
        with patch.object(sys, 'argv', argv), patch.object(provider, 'HTTPServer') as server, contextlib.redirect_stdout(output):
            with self.assertRaisesRegex(RuntimeError, '--rag-script'):
                provider.main()
        server.assert_not_called()
        self.native.assert_not_called()
        self.assertNotIn('listening', output.getvalue())

    def test_cli_configured_backend_is_initialized_before_binding(self):
        argv = ['provider', '--model', str(self.model), '--adapter', str(self.adapter), '--rag-script', str(self.backend), '--port', '0']
        output = io.StringIO()
        with patch.object(sys, 'argv', argv), patch.object(provider, 'HTTPServer') as server, contextlib.redirect_stdout(output):
            self.assertEqual(provider.main(), 0)
        server.assert_called_once_with(('127.0.0.1', 0), provider.Handler)
        self.assertEqual(server.return_value.runtime._rag.build_count, 1)
        self.assertEqual(server.return_value.runtime._collection, {'fixture': True})
        server.return_value.serve_forever.assert_called_once()
        self.assertIn('listening', output.getvalue())


if __name__ == '__main__':
    unittest.main()
