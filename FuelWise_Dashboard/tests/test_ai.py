import json
import os
import unittest
import urllib.error
from io import BytesIO
from unittest.mock import patch

from server import (
    DEFAULT_GEMINI_MODEL,
    ai_messages,
    gemini_error_message,
    gemini_model,
    request_gemini,
    sanitize_ai_context,
)


class AiContextTests(unittest.TestCase):
    def setUp(self):
        self.context = {
            'threshold_kmh_s': 3,
            'current': {
                'offset_sec': 15,
                'speed': 42,
                'rpm': 1600,
                'load': 48,
                'temp': 82,
                'battery': 27.5,
                'fuel_level': 70,
                'distance': 12,
                'used': 1.4,
                'lat': 23.5,
                'lon': 120.9,
                'plate': 'PRIVATE-PLATE',
                'vehicle': 'PRIVATE-VEHICLE',
            },
            'recent_points': [
                {'offset_sec': index, 'speed': index, 'lat': 23.5}
                for index in range(25)
            ],
            'trip_summary': {
                'observations': 100,
                'acceleration_events': 2,
                'braking_events': 1,
                'max_acceleration_kmh_s': 5.2,
                'max_braking_kmh_s': 4.1,
                'recent_events': [
                    {
                        'type': 'acceleration',
                        'start_offset_sec': 5,
                        'end_offset_sec': 8,
                        'max_kmh_s': 5.2,
                        'vehicle': 'PRIVATE-VEHICLE',
                    }
                ],
            },
        }

    def test_context_keeps_only_anonymous_sensor_fields(self):
        messages = ai_messages('有沒有急加速？', [], self.context)
        prompt = messages['contents'][-1]['parts'][0]['text']
        self.assertIn('"speed": 42.0', prompt)
        self.assertNotIn('PRIVATE-PLATE', prompt)
        self.assertNotIn('PRIVATE-VEHICLE', prompt)
        self.assertNotIn('"lat"', prompt)
        self.assertNotIn('"lon"', prompt)

    def test_recent_sensor_points_are_limited(self):
        sanitized = sanitize_ai_context(self.context)
        self.assertEqual(len(sanitized['recent_points']), 20)
        self.assertEqual(sanitized['trip_summary']['acceleration_events'], 2)

    def test_rejects_empty_or_oversized_questions(self):
        with self.assertRaises(ValueError):
            ai_messages(' ', [], self.context)
        with self.assertRaises(ValueError):
            ai_messages('x' * 1001, [], self.context)

    def test_gemini_request_uses_server_key_and_returns_model_text(self):
        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

            def read(self):
                return json.dumps(
                    {'candidates': [{'content': {'parts': [{'text': '分析完成'}]}}]},
                    ensure_ascii=False,
                ).encode()

        with (
            patch.dict(os.environ, {'GEMINI_API_KEY': 'test-only-key'}),
            patch('server.urllib.request.urlopen', return_value=FakeResponse()) as urlopen,
        ):
            answer = request_gemini(
                {
                    'system_instruction': {'parts': [{'text': 'system'}]},
                    'contents': [{'role': 'user', 'parts': [{'text': 'test'}]}],
                    'generation_config': {'temperature': 0.3, 'max_output_tokens': 500},
                }
            )

        self.assertEqual(answer, '分析完成')
        request = urlopen.call_args.args[0]
        self.assertIn('generativelanguage.googleapis.com/v1beta/models/gemini-3.8-flash:generateContent', request.full_url)
        self.assertEqual(request.get_header('X-goog-api-key'), 'test-only-key')
        self.assertEqual(json.loads(request.data)['contents'][0]['role'], 'user')

    def test_gemini_request_rejects_invalid_model_names(self):
        with (
            patch.dict(os.environ, {'GEMINI_API_KEY': 'test-only-key', 'GEMINI_MODEL': '../other'}),
            self.assertRaises(ValueError),
        ):
            request_gemini({'contents': []})

    def test_gemini_model_defaults_to_38_and_accepts_37(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(gemini_model(), DEFAULT_GEMINI_MODEL)
        with patch.dict(os.environ, {'GEMINI_MODEL': 'gemini-3.7-flash'}):
            self.assertEqual(gemini_model(), 'gemini-3.7-flash')

    def test_gemini_http_error_includes_provider_diagnostic(self):
        error = urllib.error.HTTPError(
            'https://generativelanguage.googleapis.com/',
            404,
            'Not Found',
            {},
            BytesIO(
                json.dumps(
                    {'error': {'message': 'Model is not available; key=test-only-key'}}
                ).encode()
            ),
        )

        message = gemini_error_message(error, 'gemini-test', 'test-only-key')

        self.assertIn('HTTP 404', message)
        self.assertIn('gemini-test', message)
        self.assertIn('Model is not available', message)
        self.assertNotIn('test-only-key', message)


if __name__ == '__main__':
    unittest.main()
