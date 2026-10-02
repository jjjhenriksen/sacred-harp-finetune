"""Real loopback HTTP requests use a fake runtime and never load MLX."""
import http.client
import io
import json
import socket
import threading
import time
import unittest
from email.message import Message
from http.server import HTTPServer
from types import SimpleNamespace
from unittest.mock import Mock, patch

from provider_fixture import load_provider

provider = load_provider()
BODY_LIMIT = 1024 * 1024


class QuietHandler(provider.Handler):
    def log_message(self, *args):
        pass
    def setup(self):
        super().setup()
        self.connection.settimeout(0.3)  # Bound even the original-code failure fixture.


class ProviderHttpTests(unittest.TestCase):
    def setUp(self):
        self.runtime = SimpleNamespace(
            search=Mock(return_value={'results': ['fictional evidence']}),
            complete=Mock(return_value={'content': 'Fixture answer', 'tool_calls': None, 'finish_reason': 'stop', 'prompt_tokens': 3}))
        self.server = HTTPServer(('127.0.0.1', 0), QuietHandler)
        self.server.runtime = self.runtime
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True)
        self.thread.start()
        self.addCleanup(self.stop_server)

    def stop_server(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)
        self.assertFalse(self.thread.is_alive())

    def request(self, body, path='/sacred-harp/search', headers=None):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=1)
        try:
            payload = body if isinstance(body, bytes) else json.dumps(body).encode()
            connection.request('POST', path, body=payload, headers=headers or {})
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def assert_bad(self, body, status=400, **kwargs):
        actual, response = self.request(body, **kwargs)
        self.assertEqual(actual, status, response)
        self.assertEqual(response['error']['type'], 'invalid_request_error')
        self.runtime.search.assert_not_called()
        self.runtime.complete.assert_not_called()
        # A malformed request must not prevent the next real HTTP request from succeeding.
        actual, _ = self.request({'query': 'Fictional query', 'top_k': 2})
        self.assertEqual(actual, 200)
        self.runtime.search.reset_mock()

    def test_malformed_json_arrays_and_scalars_are_client_errors(self):
        for payload in [b'{', b'\xff', '{}'.encode('utf-16'), [], None, 1, 'query', b'']:
            with self.subTest(payload=payload):
                self.assert_bad(payload)

    def test_invalid_query_and_top_k_are_client_errors(self):
        for query in [None, [], {}, 12, ' ']:
            with self.subTest(query=query):
                self.assert_bad({'query': query})
        for top_k in ['three', '2', True, False, 1.5, None, 0, -1, 6]:
            with self.subTest(top_k=top_k):
                self.assert_bad({'query': 'Fixture query', 'top_k': top_k})

    def test_invalid_lengths_are_rejected_without_reading_a_body(self):
        for length in ['-1', 'nope', '1.5', '+2']:
            with self.subTest(length=length):
                self.assert_bad(b'', headers={'Content-Length': length})

    def test_oversized_declared_bodies_are_rejected_before_upload(self):
        for length in [str(BODY_LIMIT + 1), '9' * 5000]:
            self.assert_bad(b'', status=413, headers={'Content-Length': length})

    def test_unsupported_transfer_encoding_is_a_client_error(self):
        self.assert_bad(b'{}', headers={'Transfer-Encoding': 'chunked'})

    def test_truncated_body_returns_client_error(self):
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=1)
        try:
            connection.request('POST', '/sacred-harp/search', body=b'{', headers={'Content-Length': '100'})
            connection.sock.shutdown(socket.SHUT_WR)
            response = connection.getresponse()
            self.assertEqual(response.status, 400)
            self.assertIn('incomplete', json.loads(response.read())['error']['message'].lower())
        finally:
            connection.close()
        self.runtime.search.assert_not_called()

    def test_stalled_body_times_out_as_a_client_error(self):
        with patch.object(provider, 'REQUEST_READ_TIMEOUT_SECONDS', 0.05, create=True):
            self.assert_bad(b'{', headers={'Content-Length': '100'})

    def test_trickled_upload_cannot_extend_total_deadline(self):
        connection = socket.create_connection(('127.0.0.1', self.server.server_port), timeout=1)
        stop = threading.Event()
        def trickle():
            while not stop.wait(0.02):
                try:
                    connection.sendall(b' ')
                except OSError:
                    return
        with patch.object(provider, 'REQUEST_READ_TIMEOUT_SECONDS', 0.06):
            connection.sendall(b'POST /sacred-harp/search HTTP/1.0\r\nContent-Length: 100\r\n\r\n{')
            thread = threading.Thread(target=trickle, daemon=True)
            thread.start()
            start = time.monotonic()
            try:
                response = http.client.HTTPResponse(connection)
                response.begin()
                self.assertEqual(response.status, 400)
                self.assertIn('timed out', json.loads(response.read())['error']['message'])
                self.assertLess(time.monotonic() - start, 0.5)
            finally:
                stop.set()
                thread.join(timeout=1)
                connection.close()
        self.runtime.search.assert_not_called()

    def test_valid_tool_history_content_parts_and_streaming(self):
        body = {'messages': [
            {'role': 'user', 'content': [{'type': 'text', 'text': 'Fixture question'}]},
            {'role': 'assistant', 'content': None, 'tool_calls': [{'id': 'fixture', 'type': 'function', 'function': {'name': 'sacred_harp_search', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': 'fixture', 'content': 'Fixture evidence'}],
            'tools': [{'type': 'function', 'function': {'name': 'sacred_harp_search', 'parameters': {'type': 'object'}}}],
            'max_completion_tokens': 32, 'model': None, 'stream': True, 'temperature': 0}
        connection = http.client.HTTPConnection('127.0.0.1', self.server.server_port, timeout=1)
        try:
            connection.request('POST', '/v1/chat/completions', body=json.dumps(body).encode())
            response = connection.getresponse()
            self.assertEqual(response.status, 200)
            self.assertEqual(response.getheader('Content-Type'), 'text/event-stream')
            frames = response.read().decode().split('\n\n')
            self.assertEqual(json.loads(frames[0][6:])['choices'][0]['delta']['content'], 'Fixture answer')
            self.assertEqual(json.loads(frames[1][6:])['choices'][0]['finish_reason'], 'stop')
            self.assertEqual(frames[2], 'data: [DONE]')
            self.runtime.complete.assert_called_once_with(body)
        finally:
            connection.close()

    def test_completion_shape_and_numeric_fields_are_validated_before_runtime(self):
        base = {'messages': [{'role': 'user', 'content': 'Fictional question'}]}
        for field, value in [('messages', 'bad'), ('messages', [None]), ('messages', [{'role': 1}]), ('messages', [{'role': {}}]), ('messages', []), ('messages', [{'role': 'user', 'content': [None]}]), ('messages', [{'role': 'assistant', 'tool_calls': [None]}]), ('messages', [{'role': 'user', 'content': {}}]), ('tools', [None]), ('tools', [{'function': 'bad'}]), ('max_tokens', 'many'), ('max_tokens', True), ('max_tokens', 0), ('max_completion_tokens', 1.5), ('temperature', 'warm'), ('temperature', float('nan')), ('temperature', float('inf')), ('temperature', -1), ('temperature', 3), ('temperature', 10 ** 1000), ('stream', 'false'), ('model', {})]:
            with self.subTest(field=field, value=value):
                self.assert_bad({**base, field: value}, path='/v1/chat/completions')

    def test_valid_completion_and_search_do_not_load_models(self):
        body = {'messages': [{'role': 'user', 'content': 'Fixture question'}], 'max_tokens': 32, 'temperature': 0.5, 'stream': False}
        status, result = self.request(body, path='/v1/chat/completions')
        self.assertEqual(status, 200)
        self.assertEqual(result['choices'][0]['message']['content'], 'Fixture answer')
        self.runtime.complete.assert_called_once_with(body)
        status, _ = self.request({'query': ' Fixture question ', 'top_k': 5})
        self.assertEqual(status, 200)
        self.runtime.search.assert_called_once_with('Fixture question', 5)


class BoundedReadTests(unittest.TestCase):
    def handler(self, lengths):
        handler = object.__new__(provider.Handler)
        handler.headers = Message()
        for length in lengths:
            handler.headers.add_header('Content-Length', length)
        handler.connection = Mock()
        handler.rfile = io.BytesIO(b'{}')
        return handler

    def test_duplicate_and_missing_lengths_are_rejected_before_read(self):
        for lengths in [[], ['2', '2']]:
            with self.subTest(lengths=lengths):
                handler = self.handler(lengths)
                handler.rfile = Mock()
                with self.assertRaises(provider.ClientRequestError):
                    handler.read_json()
                handler.rfile.read1.assert_not_called()

    def test_oversized_length_does_not_read_any_payload(self):
        handler = self.handler([str(BODY_LIMIT + 1)])
        handler.rfile = Mock()
        with self.assertRaises(provider.ClientRequestError) as failure:
            handler.read_json()
        self.assertEqual(getattr(failure.exception, 'status', None), 413)
        handler.rfile.read1.assert_not_called()

    def test_total_read_deadline_is_enforced_between_chunks(self):
        handler = self.handler(['3'])
        handler.rfile = SimpleNamespace(read1=lambda size: b'{}')
        with patch.object(provider.time, 'monotonic', side_effect=[0, 0, 6]):
            with self.assertRaises(provider.ClientRequestError) as failure:
                handler.read_json()
        self.assertEqual(getattr(failure.exception, 'status', None), 400)
        self.assertIn('timed out', str(failure.exception))

    def test_each_read_is_bounded_even_for_an_at_limit_body(self):
        body = json.dumps({'query': 'x' * (BODY_LIMIT - len(b'{"query": ""}'))}).encode()
        self.assertEqual(len(body), BODY_LIMIT)
        handler = self.handler([str(len(body))])
        reader = io.BytesIO(body)
        sizes = []
        def read(size):
            sizes.append(size)
            return reader.read(size)
        handler.rfile = SimpleNamespace(read1=read)
        self.assertEqual(len(handler.read_json()['query']), BODY_LIMIT - len(b'{"query": ""}'))
        self.assertTrue(sizes)
        self.assertLessEqual(max(sizes), 64 * 1024)

    def test_overproducing_reader_is_rejected_during_reading(self):
        handler = self.handler(['2'])
        handler.rfile = SimpleNamespace(read1=lambda size: b'x' * (BODY_LIMIT + 1))
        with self.assertRaises(provider.ClientRequestError) as failure:
            handler.read_json()
        self.assertEqual(getattr(failure.exception, 'status', None), 413)


if __name__ == '__main__':
    unittest.main()
