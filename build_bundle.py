"""Create an additive minimal LiftPoint deployment bundle. Never uploads or fits.

Use only after the account owner has authorized the destination's data scope.
This command preserves source bytes, arrays, customer order and all experiments.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import shutil
import sys
import tempfile
import zipfile
import numpy as np


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source = args.source_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise ValueError('Output bundle exists; preserve it and choose a new output name')
    sys.path.insert(0, str(source / 'streamlit_demo'))
    # Read the current audited local app through an isolated source name to avoid
    # accidentally resolving this folder's portable artifacts module.
    import importlib.util
    spec = importlib.util.spec_from_file_location('liftpoint_local_evidence_for_export', source / 'streamlit_demo/artifacts.py')
    local = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = local
    spec.loader.exec_module(local)
    config = local.load_config()
    replay = local.load_replay(config)
    paths = local.artifact_paths(config)
    freeze = paths['freeze']
    feature_hash = replay.meta['feature_order_sha256']
    customer_hash = replay.meta['customer_order_sha256']
    assert replay.ids.shape == (40008,) and replay.static.shape == (40008, 91)
    assert all(np.isfinite(values).all() for values in replay.summaries.values())
    # Independently compare the ordered final cohort to the frozen split, while
    # exporting no train/validation identifiers or outcomes/assignment.
    with Path(freeze['split']['path']).open(encoding='utf-8-sig', newline='') as stream:
        expected = sorted(row['client_id'] for row in csv.DictReader(stream) if row['split'] == 'test')
    assert np.array_equal(replay.ids, np.asarray(expected))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='liftpoint-export-', dir=output.parent) as temporary:
        root = Path(temporary)
        files = {}
        def copy_file(origin, relative, purpose):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(origin, target)
            files[relative] = {'sha256': sha256(target), 'bytes': target.stat().st_size, 'purpose': purpose,
                               'source_sha256': sha256(origin), 'method': 'unchanged bytes'}
        def array_file(relative, values, purpose, source_path):
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(target, **values)
            with np.load(target, allow_pickle=False) as result:
                assert set(result.files) == set(values)
                assert all(np.array_equal(result[name], values[name]) and result[name].dtype == values[name].dtype for name in values)
            files[relative] = {'sha256': sha256(target), 'bytes': target.stat().st_size, 'purpose': purpose,
                               'source_sha256': sha256(source_path), 'method': 'lossless field projection; no transform/refit',
                               'columns': list(values)}
        array_file('inputs/customers.npz', {'client_id': replay.ids, 'static': replay.static},
                   'same 40008 internal IDs/order and fixed static91 input; target/treatment excluded', paths['customers'])
        prediction_values = {'client_id': replay.ids}
        for kind, prefix in [('DL', 'dl'), ('ML', 'ml')]:
            prediction_values[prefix + '_p0'] = replay.predictions[kind][:, 0]
            prediction_values[prefix + '_p1'] = replay.predictions[kind][:, 1]
        array_file('predictions/final_predictions.npz', prediction_values,
                   'saved final p0/p1 probabilities; no outcomes or assignment', paths['predictions'])
        copy_file(paths['summary'], 'summaries/summaries.npz', 'five original-unit display summaries for all fixed customers')
        copy_file(config.experiment_root / 'freeze_manifest.json', 'evidence/freeze_manifest.json', 'immutable experiment freeze and model hashes')
        copy_file(paths['preprocessing'], 'evidence/preprocessing_manifest.json', 'unchanged final training preprocessing state and feature order')
        copy_file(paths['input_dir'] / 'manifest.json', 'evidence/test_input_manifest.json', 'fixed cohort/input provenance')
        copy_file(paths['output_dir'] / 'results.json', 'evidence/results.json', 'existing fixed aggregate and source prediction hash')
        copy_file(paths['summary_manifest'], 'evidence/summary_manifest.json', 'original-unit summary extraction audit')
        copy_file(freeze['dl']['checkpoint_path'], 'models/dl.keras', 'frozen MLP baseline seed42 checkpoint')
        for head in ('p0', 'p1'):
            copy_file(freeze['ml']['model_paths'][head], 'models/model_' + head + '.joblib', 'frozen LR C0.1 ' + head)
        for name in ('test_scorecard.csv', 'bootstrap_results.json', 'bootstrap_intervals.csv'):
            copy_file(paths['output_dir'] / name, 'aggregates/' + name, 'existing final aggregate only; never recomputed')
        for name in ('model_objects.py', 'runtime.py'):
            copy_file(config.experiment_root / 'support' / name, 'support/' + name, 'original frozen inference loader/predictor source')
        manifest = {'version': 'liftpoint-portable-v1', 'experiment_commit': local.EXPERIMENT_COMMIT,
                    'local_app_commit': '9a155396dc144d009d05c168a131ee90c8b1fc6c', 'freeze_sha256': local.FREEZE_SHA256,
                    'customer_count': 40008, 'selected_count': 12002, 'feature_count': 91,
                    'feature_order_sha256': feature_hash, 'customer_order_sha256': customer_hash,
                    'static_array_sha256': hashlib.sha256(replay.static.tobytes()).hexdigest(),
                    'prediction_array_sha256': {name: hashlib.sha256(value.tobytes()).hexdigest() for name, value in prediction_values.items() if name != 'client_id'},
                    'source_split_sha256': freeze['split']['sha256'], 'source_predictions_sha256': local.sha256(paths['predictions']),
                    'source_customers_sha256': local.sha256(paths['customers']), 'preprocessing_refit': False,
                    'outcomes_or_assignment_exported': False, 'original_raw_rows_exported': 0,
                    'new_model_input_values': 0, 'files': files}
        write_json(root / 'manifest.json', manifest)
        with zipfile.ZipFile(output, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(root.rglob('*')):
                if path.is_file():
                    archive.write(path, path.relative_to(root).as_posix())
        inventory = {'status': 'prepared_locally_not_uploaded', 'bundle_sha256': sha256(output), 'bundle_bytes': output.stat().st_size,
                     'member_count': len(files) + 1, 'customer_count': 40008, 'selected_count': 12002,
                     'feature_count': 91, 'source_projection_array_equality': True,
                     'outcomes_assignment_raw_rows_exported': False, 'files': files}
        write_json(output.with_suffix('.inventory.json'), inventory)
        write_json(output.with_suffix('.config.json'), {'bundle_path': str(output), 'bundle_sha256': inventory['bundle_sha256']})
    print(json.dumps({name: inventory[name] for name in ('status', 'bundle_sha256', 'bundle_bytes', 'member_count', 'source_projection_array_equality')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
