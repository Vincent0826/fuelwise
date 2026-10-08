import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analysis.prepare_dashboard_model import (
    REQUIRED_BUNDLE_FILES,
    bundle_matches_index,
    index_signature,
    prepare_dashboard_model,
)


class PrepareDashboardModelTests(unittest.TestCase):
    def test_bundle_matches_only_the_same_index_and_complete_artifacts(self):
        signature = {
            'source_name': 'data.xlsx',
            'source_rows': 500,
            'source_skipped_rows': 0,
            'index_size_bytes': 2048,
            'index_mtime_ns': 12345,
        }
        with tempfile.TemporaryDirectory() as directory:
            bundle_dir = Path(directory)
            (bundle_dir / 'bundle.json').write_text(
                json.dumps({'source_index_signature': signature}),
                encoding='utf-8',
            )
            for name in REQUIRED_BUNDLE_FILES[1:]:
                (bundle_dir / name).write_text('artifact', encoding='utf-8')

            self.assertTrue(bundle_matches_index(bundle_dir, signature))
            changed = {**signature, 'index_mtime_ns': 67890}
            self.assertFalse(bundle_matches_index(bundle_dir, changed))
            (bundle_dir / 'model.json').unlink()
            self.assertFalse(bundle_matches_index(bundle_dir, signature))

    def test_index_signature_tracks_the_local_index_and_source_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / 'index.sqlite'
            index_path.write_bytes(b'local-index')
            signature = index_signature(
                index_path,
                {'source': 'data.xlsx', 'rows': 20, 'skipped_rows': 1},
            )
            self.assertEqual(signature['source_name'], 'data.xlsx')
            self.assertEqual(signature['source_rows'], 20)
            self.assertEqual(signature['source_skipped_rows'], 1)
            self.assertEqual(signature['index_size_bytes'], len(b'local-index'))

    def test_matching_bundle_skips_retraining(self):
        signature = {
            'source_name': 'data.xlsx',
            'source_rows': 500,
            'source_skipped_rows': 0,
            'index_size_bytes': 2048,
            'index_mtime_ns': 12345,
        }
        with tempfile.TemporaryDirectory() as directory:
            index_path = Path(directory) / 'index.sqlite'
            index_path.write_bytes(b'index')
            bundle_dir = Path(directory) / 'bundle'
            bundle_dir.mkdir()
            (bundle_dir / 'bundle.json').write_text(
                json.dumps({'source_index_signature': signature}),
                encoding='utf-8',
            )
            for name in REQUIRED_BUNDLE_FILES[1:]:
                (bundle_dir / name).write_text('artifact', encoding='utf-8')
            with (
                patch('analysis.prepare_dashboard_model.BUNDLE_DIR', bundle_dir),
                patch('analysis.prepare_dashboard_model.index_signature', return_value=signature),
                patch('analysis.prepare_dashboard_model.train_fuel_model.run_pipeline') as train,
            ):
                path, trained = prepare_dashboard_model(
                    index_path,
                    {'source': 'data.xlsx', 'rows': 500, 'skipped_rows': 0},
                )
            self.assertEqual(path, bundle_dir)
            self.assertFalse(trained)
            train.assert_not_called()


if __name__ == '__main__':
    unittest.main()
