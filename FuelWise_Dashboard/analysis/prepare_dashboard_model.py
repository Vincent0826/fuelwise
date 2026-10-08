"""Train a local dashboard model from the already-imported SQLite index."""

import json
import shutil
from pathlib import Path

from analysis import train_fuel_model, vehicle_history_experiment

BASE = Path(__file__).resolve().parent.parent
BUNDLE_DIR = BASE / 'analysis_output' / 'experiments' / 'dashboard_model'
EXPERIMENT_ROOT = BASE / 'analysis_output' / 'experiments'
REQUIRED_BUNDLE_FILES = (
    'bundle.json',
    'model.json',
    'frozen_training_history.json',
    'trip_features_with_vehicle_history.csv',
)


def index_signature(index_path, index_info):
    stat = Path(index_path).stat()
    return {
        'source_name': index_info.get('source'),
        'source_rows': index_info.get('rows'),
        'source_skipped_rows': index_info.get('skipped_rows'),
        'index_size_bytes': stat.st_size,
        'index_mtime_ns': stat.st_mtime_ns,
    }


def bundle_matches_index(bundle_dir, signature):
    bundle_dir = Path(bundle_dir)
    if any(not (bundle_dir / name).is_file() for name in REQUIRED_BUNDLE_FILES):
        return False
    try:
        metadata = json.loads(
            (bundle_dir / 'bundle.json').read_text(encoding='utf-8')
        )
    except (OSError, json.JSONDecodeError):
        return False
    return metadata.get('source_index_signature') == signature


def _api_metric(metric):
    return {
        'n': metric['n'],
        'mae': metric['mae_l_per_100km'],
        'rmse': metric['rmse_l_per_100km'],
        'r2': metric['r2'],
        'median_absolute_error': metric['median_absolute_error_l_per_100km'],
        'p90_absolute_error': metric['p90_absolute_error_l_per_100km'],
        'equal_weight_vehicle_mae': metric[
            'equal_weight_vehicle_mae_l_per_100km'
        ],
        'per_vehicle': [
            {
                'vehicle': row['vehicle'],
                'sample_count': row['sample_count'],
                'mae': row['mae_l_per_100km'],
            }
            for row in metric['per_vehicle']
        ],
    }


def _save_bundle(history_run, baseline_run, evaluation, signature, xgboost_version):
    method = evaluation['selected_method']
    experiment_metadata = json.loads(
        (history_run / 'experiment_config.json').read_text(encoding='utf-8')
    )
    original_metadata = json.loads(
        (baseline_run / 'metadata.json').read_text(encoding='utf-8')
    )
    original_features = original_metadata['input_features']
    model_features = original_features
    if method == 'history_xgboost':
        model_features = original_features + list(
            experiment_metadata['fixed_conditions']['additional_features']
        )

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    temporary = BUNDLE_DIR / 'model.json.tmp'
    shutil.copyfile(history_run / 'selected_model.json', temporary)
    temporary.replace(BUNDLE_DIR / 'model.json')
    for name in ('frozen_training_history.json', 'trip_features_with_vehicle_history.csv'):
        temporary = BUNDLE_DIR / f'{name}.tmp'
        shutil.copyfile(history_run / name, temporary)
        temporary.replace(BUNDLE_DIR / name)

    validation = evaluation['validation']
    comparison = {
        'vehicle_median_baseline': validation['vehicle_median_baseline'][
            'mae_l_per_100km'
        ],
        'original_xgboost': validation['original_xgboost'][
            'mae_l_per_100km'
        ],
        'history_xgboost': validation['history_xgboost'][
            'mae_l_per_100km'
        ],
    }
    metadata = {
        'model_method': method,
        'model_target': 'l_per_100km',
        'model_file': 'model.json',
        'prediction_tree_count': (
            evaluation['history_model']['prediction_tree_count']
            if method == 'history_xgboost'
            else evaluation['baseline_model']['prediction_tree_count']
        ),
        'feature_names': model_features,
        'xgboost_version': xgboost_version,
        'model_role': 'experimental_trip_completion_estimate',
        'validation_metrics': {
            'l_per_100km': _api_metric(validation[method]),
        },
        'comparison_validation_mae_l_per_100km': comparison,
        'test_follow_up_metrics': {
            'l_per_100km': _api_metric(evaluation['test_follow_up'][method]),
        },
        'selection_reason': experiment_metadata['model_selection']['reason'],
        'shap_method': 'XGBoost exact TreeSHAP',
        'shap_background': 'Path-dependent',
        'source_index_signature': signature,
        'source_indexed_trip_count': original_metadata['indexed_trip_count'],
        'split_counts': original_metadata['split_counts_and_ranges'],
    }
    temporary = BUNDLE_DIR / 'bundle.json.tmp'
    temporary.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8',
    )
    temporary.replace(BUNDLE_DIR / 'bundle.json')
    return BUNDLE_DIR


def prepare_dashboard_model(index_path, index_info):
    """Train only when the local model package is absent or its index changed."""
    signature = index_signature(index_path, index_info)
    if bundle_matches_index(BUNDLE_DIR, signature):
        return BUNDLE_DIR, False

    print(
        '正在以本機資料建立行程油耗實驗模型；首次啟動可能需要數分鐘…',
        flush=True,
    )
    baseline_run, _ = train_fuel_model.run_pipeline(
        index_path=index_path
    )
    history_run, evaluation = vehicle_history_experiment.run_experiment(
        baseline_run=baseline_run,
        output_root=EXPERIMENT_ROOT,
    )
    try:
        import xgboost
    except ImportError as exc:
        raise RuntimeError(
            '模型訓練需要 XGBoost；請重新執行啟動程式安裝 requirements.txt。'
        ) from exc

    bundle = _save_bundle(
        history_run,
        baseline_run,
        evaluation,
        signature,
        xgboost.__version__,
    )
    return bundle, True


if __name__ == '__main__':
    index_path, index_info = train_fuel_model.find_existing_index()
    path, trained = prepare_dashboard_model(index_path, index_info)
    print(f"{'已建立' if trained else '已存在'}行程油耗模型：{path}", flush=True)
