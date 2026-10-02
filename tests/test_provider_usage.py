"""Actual handler responses exercise runtime generation with deterministic metadata."""
import contextlib
import http.client
import json
import threading
import unittest
from http.server import HTTPServer
from types import SimpleNamespace
from unittest.mock import Mock, patch

from provider_fixture import load_provider

provider = load_provider()


class QuietHandler(provider.Handler):
    def log_message(self, *args):
        pass


class ProviderUsageTests(unittest.TestCase):
    def setUp(self):
        self.runtime = object.__new__(provider.SpecialistRuntime)
        self.runtime.model = object()
        self.runtime.tokenizer = Mock()
        self.runtime.tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5, 6]
        self.runtime.lock = threading.Lock()
        self.server = HTTPServer(('127.0.0.1', 0), QuietHandler)
        self.server.runtime = self.runtime
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)
        self.body = {'messages': [{'role': 'user', 'content': 'Fictional plain question'}]}

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())

    def request(self, body=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=1)
        try:
            connection.request('POST', '/v1/chat/completions', body=json.dumps(body or self.body).encode())
            response = connection.getresponse()
            result = json.loads(response.read())
            self.assertEqual(response.status, 200, result)
            return result
        finally:
            connection.close()

    def generation(self, responses, fallback=None, parsed=None):
        # The old string-only generator is patched too so baseline assertions reach HTTP usage.
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        self.stream = stack.enter_context(patch.object(provider, 'stream_generate', return_value=iter(responses), create=True))
        stack.enter_context(patch.object(provider, 'generate', return_value=''.join(part.text for part in responses), create=True))
        stack.enter_context(patch.object(provider, 'make_sampler', return_value=object()))
        stack.enter_context(patch.object(provider, 'interpretive_fallback_answer', return_value=fallback))
        stack.enter_context(patch.object(provider, 'parse_tool_call', return_value=parsed))

    def test_http_reports_final_generation_counters_not_characters_or_chunks(self):
        responses = [SimpleNamespace(text=' Fixture', prompt_tokens=3, generation_tokens=2),
                     SimpleNamespace(text=' answer ', prompt_tokens=3, generation_tokens=8)]
        self.generation(responses)
        result = self.request()
        self.assertEqual(result['choices'][0]['message']['content'], 'Fixture answer')
        self.assertEqual(result['usage'], {'prompt_tokens': 3, 'completion_tokens': 8, 'total_tokens': 11})
        self.stream.assert_called_once()
        self.assertEqual(self.stream.call_args.args[2], [1, 2, 3, 4, 5, 6])

    def test_generated_counts_survive_answer_fallback(self):
        self.generation([SimpleNamespace(text='x', prompt_tokens=4, generation_tokens=2)], fallback='Much longer grounded replacement answer')
        result = self.request()
        self.assertEqual(result['choices'][0]['message']['content'], 'Much longer grounded replacement answer')
        self.assertEqual(result['usage'], {'prompt_tokens': 4, 'completion_tokens': 2, 'total_tokens': 6})

    def test_generated_tool_call_preserves_counts(self):
        self.generation([SimpleNamespace(text='function', prompt_tokens=9, generation_tokens=12)], parsed=('fixture_tool', {'value': 1}))
        result = self.request()
        self.assertEqual(result['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(result['choices'][0]['message']['tool_calls'][0]['function']['name'], 'fixture_tool')
        self.assertEqual(result['usage'], {'prompt_tokens': 9, 'completion_tokens': 12, 'total_tokens': 21})

    def test_final_empty_text_still_uses_final_counters(self):
        self.generation([SimpleNamespace(text='Answer', prompt_tokens=5, generation_tokens=1),
                         SimpleNamespace(text='', prompt_tokens=5, generation_tokens=2)])
        result = self.request()
        self.assertEqual(result['choices'][0]['message']['content'], 'Answer')
        self.assertEqual(result['usage']['completion_tokens'], 2)

    def test_missing_invalid_or_empty_generation_metadata_omits_usage(self):
        cases = [[], [SimpleNamespace(text='Answer')],
                 [SimpleNamespace(text='Answer', prompt_tokens=4)],
                 [SimpleNamespace(text='Answer', prompt_tokens=True, generation_tokens=3)],
                 [SimpleNamespace(text='Answer', prompt_tokens=4, generation_tokens=-1)],
                 [SimpleNamespace(text='Answer', prompt_tokens=4, generation_tokens=1), SimpleNamespace(text=' more')]]
        for responses in cases:
            with self.subTest(responses=responses):
                self.generation(responses)
                self.assertNotIn('usage', self.request())

    def test_deterministic_route_omits_model_usage_and_does_not_generate(self):
        body = {'messages': [{'role': 'user', 'content': 'Scan the last ten messages'}],
                'tools': [{'type': 'function', 'function': {'name': 'discussion', 'parameters': {'type': 'object'}}}]}
        self.generation([])
        result = self.request(body)
        self.assertEqual(result['choices'][0]['finish_reason'], 'tool_calls')
        self.assertNotIn('usage', result)
        self.stream.assert_not_called()
        self.runtime.tokenizer.apply_chat_template.assert_not_called()

    def test_direct_grounded_answer_omits_model_usage(self):
        body = {'messages': [{'role': 'user', 'content': 'Fixture question'}, {'role': 'tool', 'content': 'Fixture evidence'}]}
        self.generation([])
        with patch.object(provider, 'direct_grounded_answer', return_value='Verbatim fixture verse'):
            result = self.request(body)
        self.assertEqual(result['choices'][0]['message']['content'], 'Verbatim fixture verse')
        self.assertNotIn('usage', result)
        self.stream.assert_not_called()

    def test_handler_omits_usage_when_runtime_has_no_metric(self):
        self.server.runtime = SimpleNamespace(complete=Mock(return_value={'content': 'Fixture answer', 'tool_calls': None, 'finish_reason': 'stop'}))
        self.assertNotIn('usage', self.request())


if __name__ == '__main__':
    unittest.main()
