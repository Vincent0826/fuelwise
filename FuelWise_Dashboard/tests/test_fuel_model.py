import unittest
import math
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis import fuel_model_inference
from analysis.train_fuel_model import (
    FEATURE_COLUMNS,
    FORBIDDEN_FEATURE_TOKENS,
    _assert_no_leakage,
    extract_features,
    temporal_split,
)
from analysis.vehicle_history_experiment import (
    HISTORY_FEATURES,
    build_vehicle_history_features,
)
from analytics import analyze


def sample_trip(start, end_offset=10, vehicle='vehicle-1', journey=None):
    begin = datetime(2025, 1, 1) + timedelta(days=start)
    return {
        'vehicle': vehicle,
        'journey': journey or f'journey-{start}',
        'start': begin.isoformat(sep=' '),
        'end': (begin + timedelta(seconds=end_offset)).isoformat(sep=' '),
        '_start_dt': begin,
        '_end_dt': begin + timedelta(seconds=end_offset),
    }


def raw_point(seconds, speed, rpm=1000, fuel=100, odo=10):
    stamp = datetime(2025, 1, 1) + timedelta(seconds=seconds)
    return {
        'time': stamp.isoformat(sep=' '),
        'can.canStatus': 0,
        'can.canSpeed': speed,
        'can.engine.rpm': rpm,
        'can.engine.totalFuelUsed': fuel,
        'can.totalMileage': odo,
        'can.engine.engineLoad': 50,
        'can.engine.engineCoolantTemp': 80,
    }


class FuelModelTests(unittest.TestCase):
    def test_training_history_uses_only_strictly_completed_prior_trips(self):
        training = [
            {
                'vehicle': 'A',
                'journey': 'first',
                'start': datetime(2025, 1, 1, 0, 0),
                'end': datetime(2025, 1, 1, 0, 10),
                'target_l_per_100km': 10.0,
            },
            {
                'vehicle': 'B',
                'journey': 'not-yet-complete',
                'start': datetime(2025, 1, 1, 0, 5),
                'end': datetime(2025, 1, 1, 0, 20),
                'target_l_per_100km': 30.0,
            },
            {
                'vehicle': 'A',
                'journey': 'current',
                'start': datetime(2025, 1, 1, 0, 11),
                'end': datetime(2025, 1, 1, 0, 30),
                'target_l_per_100km': 900.0,
            },
        ]
        features = build_vehicle_history_features(training, training)
        current = features[2]
        self.assertEqual(current[HISTORY_FEATURES[0]], 10.0)
        self.assertEqual(current[HISTORY_FEATURES[1]], 1)
        self.assertEqual(current[HISTORY_FEATURES[2]], 0)

        changed_current = [dict(row) for row in training]
        changed_current[2]['target_l_per_100km'] = 1.0
        changed_features = build_vehicle_history_features(
            changed_current,
            changed_current,
        )
        self.assertEqual(features[2], changed_features[2])

    def test_frozen_evaluation_history_ignores_validation_and_test_targets(self):
        training = [
            {
                'vehicle': 'A',
                'journey': 'train-a',
                'start': datetime(2025, 1, 1),
                'end': datetime(2025, 1, 1, 0, 10),
                'target_l_per_100km': 12.0,
            },
            {
                'vehicle': 'A',
                'journey': 'train-b',
                'start': datetime(2025, 1, 2),
                'end': datetime(2025, 1, 2, 0, 10),
                'target_l_per_100km': 20.0,
            },
        ]
        evaluation = [
            {
                'vehicle': 'A',
                'journey': 'validation',
                'start': datetime(2025, 1, 3),
                'end': datetime(2025, 1, 3, 0, 10),
                'target_l_per_100km': 500.0,
            },
            {
                'vehicle': 'UNSEEN',
                'journey': 'test',
                'start': datetime(2025, 1, 4),
                'end': datetime(2025, 1, 4, 0, 10),
                'target_l_per_100km': 700.0,
            },
        ]
        first = build_vehicle_history_features(
            training,
            evaluation,
            frozen=True,
        )
        changed_evaluation = [dict(row) for row in evaluation]
        changed_evaluation[0]['target_l_per_100km'] = 1.0
        changed_evaluation[1]['target_l_per_100km'] = 2.0
        second = build_vehicle_history_features(
            training,
            changed_evaluation,
            frozen=True,
        )
        self.assertEqual(first, second)
        self.assertEqual(first[0][HISTORY_FEATURES[0]], 16.0)
        self.assertEqual(first[0][HISTORY_FEATURES[1]], 2)
        self.assertEqual(first[0][HISTORY_FEATURES[2]], 0)
        self.assertEqual(first[1][HISTORY_FEATURES[0]], 16.0)
        self.assertEqual(first[1][HISTORY_FEATURES[1]], 0)
        self.assertEqual(first[1][HISTORY_FEATURES[2]], 1)

    def test_history_features_keep_missing_fallback_nan(self):
        row = {
            'vehicle': 'A',
            'journey': 'only-trip',
            'start': datetime(2025, 1, 1),
            'end': datetime(2025, 1, 1, 0, 10),
            'target_l_per_100km': 12.0,
        }
        features = build_vehicle_history_features([row], [row])[0]
        self.assertTrue(math.isnan(features[HISTORY_FEATURES[0]]))
        self.assertEqual(features[HISTORY_FEATURES[1]], 0)
        self.assertEqual(features[HISTORY_FEATURES[2]], 1)

    def test_saved_training_trip_uses_its_causal_history_features(self):
        saved_values = {
            name: 1.0 for name in FEATURE_COLUMNS + HISTORY_FEATURES
        }
        saved_values[HISTORY_FEATURES[0]] = 12.5
        saved_values[HISTORY_FEATURES[1]] = 2
        bundle = {
            'saved_features': {
                ('A', 'train-trip'): {
                    'split': 'train',
                    'features': saved_values,
                }
            },
            'history': {
                'global_training_median_l_per_100km': 99.0,
                'training_vehicle_history': {
                    'A': {
                        'median_l_per_100km': 80.0,
                        'completed_training_trip_count': 20,
                    }
                },
            },
        }
        values, split, source = fuel_model_inference._feature_values(
            {},
            'A',
            'train-trip',
            bundle,
        )
        self.assertEqual(values[HISTORY_FEATURES[0]], 12.5)
        self.assertEqual(values[HISTORY_FEATURES[1]], 2)
        self.assertEqual(split, 'train')
        self.assertEqual(source, 'saved_trip_features')

    def test_tree_shap_additivity_and_model_reload_prediction(self):
        import xgboost as xgb

        feature_names = ['distance_km', 'duration_minutes']
        features = pd.DataFrame(
            [[1.0, 2.0], [2.0, 3.0], [5.0, 7.0], [9.0, 10.0]],
            columns=feature_names,
        )
        matrix = xgb.DMatrix(
            features,
            label=[2.0, 4.0, 8.0, 14.0],
            feature_names=feature_names,
        )
        model = xgb.train(
            {
                'objective': 'reg:squarederror',
                'max_depth': 2,
                'eta': 0.3,
                'seed': 17,
                'nthread': 1,
            },
            matrix,
            num_boost_round=4,
        )
        values = {'distance_km': 4.0, 'duration_minutes': np.inf}
        before = fuel_model_inference._predict_and_explain(
            model,
            values,
            feature_names,
            4,
        )
        self.assertTrue(math.isfinite(before[1]))
        self.assertLess(abs(before[4]), 1e-5)

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / 'model.json'
            model.save_model(model_path)
            reloaded = xgb.Booster()
            reloaded.load_model(model_path)
            after = fuel_model_inference._predict_and_explain(
                reloaded,
                values,
                feature_names,
                4,
            )
        self.assertAlmostEqual(before[1], after[1], places=7)
        np.testing.assert_allclose(before[3], after[3], rtol=0, atol=1e-7)

    def test_fuel_saving_scenario_uses_shap_and_training_q25(self):
        import xgboost as xgb

        feature_name = 'mean_abs_acceleration_m_s2'
        feature_names = [feature_name]
        training_values = np.arange(8, dtype=float)
        training = pd.DataFrame({feature_name: training_values})
        matrix = xgb.DMatrix(
            training,
            label=training_values * 2 + 5,
            feature_names=feature_names,
        )
        model = xgb.train(
            {
                'objective': 'reg:squarederror',
                'max_depth': 1,
                'eta': 0.5,
                'seed': 31,
                'nthread': 1,
            },
            matrix,
            num_boost_round=4,
        )
        values = {feature_name: 7.0}
        _, prediction, _, shap_values, _ = fuel_model_inference._predict_and_explain(
            model,
            values,
            feature_names,
            4,
        )
        bundle = {
            'saved_features': {
                (f'vehicle-{index}', f'trip-{index}'): {
                    'split': 'train',
                    'features': {feature_name: float(value)},
                }
                for index, value in enumerate(training_values)
            }
        }

        scenarios = fuel_model_inference._fuel_saving_scenarios(
            model,
            values,
            shap_values,
            feature_names,
            4,
            prediction,
            100.0,
            'l_per_100km',
            bundle,
        )

        self.assertEqual(len(scenarios), 1)
        self.assertEqual(scenarios[0]['key'], feature_name)
        self.assertAlmostEqual(
            scenarios[0]['benchmark_value'],
            np.quantile(training_values, 0.25),
        )
        self.assertGreater(scenarios[0]['shap_contribution'], 0)
        self.assertGreater(scenarios[0]['estimated_saving_pct'], 0)
        self.assertGreater(
            scenarios[0]['estimated_saving_l_per_100km'],
            0,
        )

    def test_predict_trip_keeps_units_and_reports_shap_values(self):
        import xgboost as xgb

        feature_names = FEATURE_COLUMNS + HISTORY_FEATURES
        training_values = np.arange(4 * len(feature_names), dtype=float).reshape(
            4,
            len(feature_names),
        )
        training = pd.DataFrame(training_values, columns=feature_names)
        matrix = xgb.DMatrix(
            training,
            label=[10.0, 20.0, 30.0, 40.0],
            feature_names=feature_names,
        )
        model = xgb.train(
            {
                'objective': 'reg:squarederror',
                'max_depth': 2,
                'eta': 0.2,
                'seed': 23,
                'nthread': 1,
            },
            matrix,
            num_boost_round=3,
        )
        values = {
            name: float(training.iloc[0][name])
            for name in feature_names
        }
        values['distance_km'] = 5.0
        values['vehicle_history_median_l_per_100km'] = 16.0
        bundle = {
            'metadata': {
                'model_method': 'history_xgboost',
                'model_target': 'l_per_100km',
                'model_role': 'experimental_trip_completion_estimate',
                'feature_names': feature_names,
                'prediction_tree_count': 3,
                'xgboost_version': '3.2.0',
                'validation_metrics': {
                    'l_per_100km': {'mae': 14.0, 'n': 100}
                },
                'comparison_validation_mae_l_per_100km': {
                    'vehicle_median_baseline': 13.0
                },
                'selection_reason': '驗證集 MAE 比原模型低。',
                'shap_method': 'XGBoost exact TreeSHAP',
                'shap_background': 'Path-dependent',
            },
            'history': {
                'global_training_median_l_per_100km': 17.0,
                'training_vehicle_history': {
                    'A': {
                        'median_l_per_100km': 30.0,
                        'completed_training_trip_count': 8,
                    }
                },
            },
            'model': model,
            'tree_count': 3,
            'saved_features': {
                ('A', 'known-trip'): {
                    'split': 'train',
                    'features': values,
                }
            },
        }
        analysis = {
            'summary': {
                'distance_km': 5.0,
                'fuel_l': 1.0,
                'l100': 20.0,
            }
        }
        with patch(
            'analysis.fuel_model_inference.load_bundle',
            return_value=bundle,
        ), patch(
            'analysis.fuel_model_inference.evaluate_applicability',
            return_value={'model_eligible': True, 'rule_version': 'test'},
        ):
            result = fuel_model_inference.predict_trip(
                analysis,
                'A',
                'known-trip',
            )
        self.assertTrue(result['available'])
        self.assertEqual(result['data_split'], 'train')
        self.assertEqual(result['history_baseline']['l_per_100km'], 16.0)
        self.assertEqual(result['explanation']['target_unit'], 'L/100 km')
        self.assertLess(abs(result['explanation']['additivity_error']), 1e-5)
        self.assertAlmostEqual(
            result['explanation']['base_value']
            + sum(item['shap_value'] for item in result['explanation']['features']),
            result['explanation']['prediction'],
            places=5,
        )

    def test_ineligible_trip_gets_no_estimate_or_shap(self):
        with patch(
            'analysis.fuel_model_inference.load_bundle',
            side_effect=AssertionError('model must not load'),
        ):
            result = fuel_model_inference.predict_trip(
                {'summary': {'distance_km': 1.0, 'duration_sec': 100}, 'points': []},
                'A',
                'short',
            )
        self.assertTrue(result['available'])
        self.assertFalse(result['eligible'])
        self.assertNotIn('explanation', result)
        self.assertNotIn('estimate', result)
        self.assertIn('short_distance', result['applicability']['labels'])

    def test_model_missing_and_invalid_distance_return_explicit_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            fuel_model_inference.load_bundle.cache_clear()
            with patch.object(
                fuel_model_inference,
                'BUNDLE_DIR',
                Path(directory),
            ):
                with self.assertRaises(
                    fuel_model_inference.ModelUnavailableError
                ):
                    fuel_model_inference.load_bundle()
            fuel_model_inference.load_bundle.cache_clear()
        eligible = {'model_eligible': True, 'rule_version': 'test'}
        with patch(
            'analysis.fuel_model_inference.evaluate_applicability',
            return_value=eligible,
        ), patch(
            'analysis.fuel_model_inference.load_bundle',
            side_effect=fuel_model_inference.ModelUnavailableError(
                '模型檔不存在'
            ),
        ):
            with self.assertRaisesRegex(
                fuel_model_inference.ModelUnavailableError,
                '模型檔不存在',
            ):
                fuel_model_inference.predict_trip(
                    {'summary': {'distance_km': 3.0}},
                    'A',
                    'trip',
                )
        with patch(
            'analysis.fuel_model_inference.evaluate_applicability',
            return_value=eligible,
        ), patch(
            'analysis.fuel_model_inference.load_bundle',
            return_value={},
        ):
            with self.assertRaisesRegex(
                fuel_model_inference.TripInferenceError,
                '有效里程不是正數',
            ):
                fuel_model_inference.predict_trip(
                    {'summary': {'distance_km': float('nan')}},
                    'A',
                    'trip',
                )

    def test_feature_inputs_exclude_target_derived_and_identifiers(self):
        _assert_no_leakage()
        for feature in FEATURE_COLUMNS:
            self.assertFalse(
                any(token in feature.lower() for token in FORBIDDEN_FEATURE_TOKENS),
                feature,
            )

    def test_temporal_split_drops_trips_crossing_boundaries(self):
        rows = [sample_trip(day, end_offset=10) for day in range(100)]
        crossing = sample_trip(69, end_offset=2 * 24 * 60 * 60, journey='crossing')
        rows.append(crossing)
        splits, excluded, first, second = temporal_split(rows)
        self.assertTrue(any(key[1] == 'crossing' for key, _ in excluded))
        sets = {
            name: {(row['vehicle'], row['journey']) for row in values}
            for name, values in splits.items()
        }
        self.assertFalse(sets['train'] & sets['validation'])
        self.assertFalse(sets['train'] & sets['test'])
        self.assertFalse(sets['validation'] & sets['test'])
        self.assertLess(
            max(row['_end_dt'] for row in splits['train']),
            min(row['_start_dt'] for row in splits['validation']),
        )
        self.assertLess(
            max(row['_end_dt'] for row in splits['validation']),
            min(row['_start_dt'] for row in splits['test']),
        )
        self.assertLess(first, second)

    def test_features_include_coverage_and_kinetic_proxy_without_fuel(self):
        result = analyze(
            [
                raw_point(0, 0),
                raw_point(10, 36, fuel=100.5, odo=10.1),
                raw_point(20, 0, fuel=101, odo=10.2),
            ],
            detail=True,
        )
        features, coverage, quality = extract_features(result)
        self.assertEqual(set(features), set(FEATURE_COLUMNS))
        self.assertEqual(set(coverage), {f'coverage_pct__{name}' for name in FEATURE_COLUMNS})
        self.assertEqual(features['kinetic_increase_j_per_kg'], 50.0)
        self.assertIsNotNone(features['kinetic_increase_j_per_kg_km'])
        self.assertFalse(any('fuel' in name.lower() for name in features))
        self.assertFalse(quality['fuel_reset'])

    def test_feature_only_analysis_preserves_existing_summary_and_quality(self):
        rows = [
            raw_point(0, 0),
            raw_point(10, 10, fuel=100.5, odo=10.1),
        ]
        full = analyze(rows, detail=True)
        compact = analyze(rows, detail=True, feature_only=True)
        self.assertEqual(compact['summary'], full['summary'])
        self.assertEqual(compact['quality'], full['quality'])
        self.assertEqual(compact['points'], full['points'])
        self.assertNotIn('segments', compact)
        self.assertNotIn('raw', compact)


if __name__ == '__main__':
    unittest.main()
