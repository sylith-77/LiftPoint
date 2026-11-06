"""Compare frozen inference to saved probabilities; never computes test performance."""
import argparse
from datetime import datetime
import json
import sys
import time
from zoneinfo import ZoneInfo
import numpy as np
from artifacts import (ATOL, RTOL, EXPERIMENT_COMMIT, FREEZE_SHA256, artifact_paths, inference_fingerprint,
                       load_config, read_json, sha256, _validate_input, _checked)
from selection import select_indices, SELECTED


def verify(config, model):
    started = time.perf_counter()
    stamp = datetime.now(ZoneInfo('Asia/Seoul')).isoformat()
    record = {'model': model, 'status': 'failed', 'checked_at_kst': stamp, 'atol': ATOL, 'rtol': RTOL,
              'experiment_commit': EXPERIMENT_COMMIT, 'freeze_sha256': FREEZE_SHA256,
              'scope': 'frozen probability/selection reproduction only; no outcome metrics, fitting, bootstrap or model selection',
              'probabilities_match': False, 'selected_set_exact_match': False}
    try:
        fingerprint = inference_fingerprint(config, model)
        record['fingerprint'] = fingerprint
        p = artifact_paths(config)
        ids, _, names, feature_hash = _validate_input(config, p)
        _checked(config, 'predictions/final_predictions.npz')
        with np.load(p['predictions'], allow_pickle=False) as saved:
            if not np.array_equal(ids, saved['client_id']):
                raise ValueError('Internal customer IDs/order differ; identifiers suppressed')
            prefix = model.lower()
            expected = np.column_stack([saved[prefix + '_p0'], saved[prefix + '_p1']])
        from inference import predict_frozen
        actual, details = predict_frozen(config, model, _verification=True)
        errors = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
        actual_selected = set(ids[select_indices(ids, actual)[:SELECTED]].tolist())
        expected_selected = set(ids[select_indices(ids, expected)[:SELECTED]].tolist())
        difference = len(actual_selected.symmetric_difference(expected_selected))
        probabilities_match = bool(np.allclose(actual, expected, atol=ATOL, rtol=RTOL))
        record.update({'customer_count': len(ids), 'customer_order_exact_match': True, 'feature_count': len(names),
                       'feature_order_sha256': feature_hash, 'customer_order_sha256': p['input_meta']['customer_order_sha256'],
                       'preprocessing_refit': False, 'target_or_treatment_loaded': False,
                       'max_probability_abs_error': float(errors.max()), 'max_p0_abs_error': float(errors[:, 0].max()),
                       'max_p1_abs_error': float(errors[:, 1].max()), 'probabilities_match': probabilities_match,
                       'selected_count_saved': len(expected_selected), 'selected_count_actual': len(actual_selected),
                       'selected_symmetric_difference': difference, 'selected_set_exact_match': difference == 0,
                       'inference_details': details})
        if inference_fingerprint(config, model)['digest'] != fingerprint['digest']:
            raise ValueError('Artifact/code/environment changed while verifying')
        if not probabilities_match:
            raise ValueError('Saved probabilities differ beyond fixed atol=1e-6, rtol=0')
        if difference:
            raise ValueError('Selected original internal ID sets differ despite probability comparison')
        record['status'] = 'passed'
    except Exception as exc:
        # No traceback or customer-bearing array is serialized to shared logs.
        record['failure_type'] = type(exc).__name__
        record['failure_reason'] = 'Deployment reproduction failed; protected values and host paths suppressed.'
    record['seconds'] = time.perf_counter() - started
    config.record_dir.mkdir(parents=True, exist_ok=True)
    name = model + '.' + datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y%m%dT%H%M%S%f') + '.json'
    path = config.record_dir / name
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    (config.record_dir / (model + '.latest.json')).write_text(json.dumps({'record_file': name, 'record_sha256': sha256(path)}, indent=2), encoding='utf-8')
    print(json.dumps({k: record[k] for k in ('model', 'status', 'atol', 'rtol', 'max_probability_abs_error', 'selected_symmetric_difference', 'seconds', 'failure_reason') if k in record}, ensure_ascii=False), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--model', choices=['all', 'DL', 'ML'], default='all')
    parser.add_argument('--aggregate-output', help='Optional public aggregate-only JSON record; no customer rows')
    args = parser.parse_args()
    config = load_config(args.config)
    records = [verify(config, model) for model in (['DL', 'ML'] if args.model == 'all' else [args.model])]
    if args.aggregate_output:
        from pathlib import Path
        Path(args.aggregate_output).write_text(json.dumps({'experiment_commit': EXPERIMENT_COMMIT, 'freeze_sha256': FREEZE_SHA256,
              'atol': ATOL, 'rtol': RTOL, 'records': records}, ensure_ascii=False, indent=2), encoding='utf-8')
    return 0 if all(r['status'] == 'passed' for r in records) else 1


if __name__ == '__main__':
    sys.exit(main())
