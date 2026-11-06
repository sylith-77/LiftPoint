"""Portable, read-only LiftPoint evidence resolver. No model-library imports.

The minimal ZIP is published in the dedicated deployment repository. Its
expected content SHA is configured independently. No customer outcome or
assignment arrays are accepted by this deployment bundle.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import zipfile
import numpy as np

EXPERIMENT_COMMIT = 'e1d52d29a7ae9145a7637b14a0c3aefc14602b56'
LOCAL_APP_COMMIT = '9a155396dc144d009d05c168a131ee90c8b1fc6c'
FREEZE_SHA256 = '7a1c42fb57861db97829ae088e375268d852bfc5e69a51bee5df66bd814b897f'
ATOL = 1e-6
RTOL = 0
SUMMARY_FIELDS = ('receipt_count', 'amount_total', 'recency_days', 'recent30_receipts', 'previous30_receipts')
CODE_DIR = Path(__file__).resolve().parent
INFERENCE_CODE = ('artifacts.py', 'inference.py', 'selection.py', 'verify_reproduction.py', 'admin_verify.py', 'streamlit_app.py')
FINAL_AGGREGATE_SHA256 = {
    'test_scorecard.csv': 'dc8f618df198022bb7e3011f331f465f52e2a154d2b88bd2442435993ff07e68',
    'bootstrap_results.json': '10636ae4c41fac1ba190668d7bcedef4245183cfb240104bdc3b0357088a18e6',
    'bootstrap_intervals.csv': '853cb4eaabc083896959ee7eb6e72662153615da89de3d1b6b36979c507cc4a4',
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _code_commit():
    explicit = os.environ.get('LIFTPOINT_CODE_COMMIT')
    if explicit:
        require(len(explicit) == 40 and all(c in '0123456789abcdef' for c in explicit), 'Invalid deployment code commit')
    try:
        result = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=CODE_DIR, capture_output=True, text=True, timeout=5, check=True)
        value = result.stdout.strip()
        if len(value) == 40 and all(c in '0123456789abcdef' for c in value):
            require(not explicit or explicit == value, 'Configured deployment commit differs from actual checkout')
            return value
    except ValueError:
        raise
    except Exception:
        pass
    return explicit or 'uncommitted-local-portability-check'


@dataclass(frozen=True)
class Config:
    bundle_root: Path
    record_dir: Path
    bundle_sha256: str
    code_commit: str
    archive_path: Path

    def payload(self):
        # Displayable metadata deliberately excludes host/user filesystem paths.
        return {'bundle_sha256': self.bundle_sha256, 'code_commit': self.code_commit,
                'layout': 'content-verified portable bundle v1', 'record_storage': 'server-local or matching Secrets record'}


def _secret_settings():
    # Streamlit is already loaded in the UI, but the standalone CLI need not load it.
    if 'streamlit' not in sys.modules:
        return {}
    try:
        import streamlit as st
        return dict(st.secrets.get('liftpoint', {}))
    except Exception:
        return {}


def _unpack(archive, expected):
    require(len(expected) == 64 and all(c in '0123456789abcdef' for c in expected), 'Protected bundle SHA-256 is required')
    require(sha256(archive) == expected, 'Protected bundle content/version mismatch')
    root = Path(os.environ.get('LIFTPOINT_CACHE_ROOT', str(Path(tempfile.gettempdir()) / 'liftpoint-protected'))) / expected
    if root.is_dir():
        require((root / 'manifest.json').is_file(), 'Protected bundle extraction is incomplete')
        with zipfile.ZipFile(archive) as zf:
            require(hashlib.sha256(zf.read('manifest.json')).hexdigest() == sha256(root / 'manifest.json'), 'Protected extraction manifest changed')
        return root
    root.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as zf:
        infos = zf.infolist()
        require(sum(i.file_size for i in infos) <= 100 * 1024**2, 'Protected bundle exceeds the minimal deployment envelope')
        names = set()
        for info in infos:
            path = PurePosixPath(info.filename)
            require(not path.is_absolute() and '..' not in path.parts and '\\' not in info.filename and not info.is_dir(), 'Invalid protected bundle path')
            require(info.filename not in names, 'Duplicate protected bundle member')
            names.add(info.filename)
        manifest = json.loads(zf.read('manifest.json').decode('utf-8'))
        require(names == {'manifest.json', *manifest['files']}, 'Protected bundle member manifest mismatch')
        for name in names:
            output = root / PurePosixPath(name)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(zf.read(name))
    manifest = read_json(root / 'manifest.json')
    for name, item in manifest['files'].items():
        require(sha256(root / name) == item['sha256'], 'Protected bundle member content mismatch')
    return root


def load_config(path=None):
    settings = _secret_settings()
    chosen = path or os.environ.get('LIFTPOINT_CONFIG')
    if chosen:
        settings.update(read_json(chosen))
    location = os.environ.get('LIFTPOINT_BUNDLE_PATH') or settings.get('bundle_path')
    expected = os.environ.get('LIFTPOINT_BUNDLE_SHA256') or settings.get('bundle_sha256')
    require(bool(location) and bool(expected), 'Protected deployment artifacts have not been configured')
    archive = Path(location).expanduser()
    if not archive.is_absolute():
        archive = Path.cwd() / archive
    root = _unpack(archive, expected)
    records = os.environ.get('LIFTPOINT_RECORD_DIR') or settings.get('record_dir')
    record_dir = Path(records).expanduser() if records else root / 'reproduction_records'
    actual_commit = _code_commit()
    configured_commit = str(settings.get('code_commit') or '')
    if configured_commit:
        require(len(configured_commit) == 40 and all(c in '0123456789abcdef' for c in configured_commit), 'Invalid configured deployment commit')
        require(actual_commit == 'uncommitted-local-portability-check' or actual_commit == configured_commit,
                'Configured deployment commit differs from actual checkout')
    code_commit = configured_commit or actual_commit
    return Config(root, record_dir, expected, code_commit, archive.resolve())


def _checked(config, relative):
    manifest = read_json(config.bundle_root / 'manifest.json')
    require(manifest['experiment_commit'] == EXPERIMENT_COMMIT and manifest['freeze_sha256'] == FREEZE_SHA256, 'Protected evidence identity mismatch')
    require(relative in manifest['files'], 'Unlisted protected evidence')
    path = config.bundle_root / relative
    require(sha256(path) == manifest['files'][relative]['sha256'], 'Protected evidence content mismatch: ' + relative)
    return path


def artifact_paths(config):
    freeze_path = _checked(config, 'evidence/freeze_manifest.json')
    require(sha256(freeze_path) == FREEZE_SHA256, 'Original freeze identity changed')
    freeze = read_json(freeze_path)
    metadata = read_json(config.bundle_root / 'manifest.json')
    train = read_json(_checked(config, 'evidence/preprocessing_manifest.json'))
    require(sha256(config.bundle_root / 'evidence/preprocessing_manifest.json') == freeze['preprocessing']['manifest_sha256'], 'Final training preprocessing changed')
    require(freeze['dl']['seed'] == 42 and freeze['ml']['configuration_id'] == 'lr_C01', 'Frozen representatives changed')
    return {'freeze': freeze, 'bundle': metadata,
            'input_meta': read_json(_checked(config, 'evidence/test_input_manifest.json')),
            'results': read_json(_checked(config, 'evidence/results.json')),
            'input_dir': config.bundle_root / 'inputs', 'output_dir': config.bundle_root / 'aggregates',
            'customers': config.bundle_root / 'inputs/customers.npz',
            'predictions': config.bundle_root / 'predictions/final_predictions.npz',
            'preprocessing': config.bundle_root / 'evidence/preprocessing_manifest.json',
            'summary': config.bundle_root / 'summaries/summaries.npz',
            'summary_manifest': config.bundle_root / 'evidence/summary_manifest.json',
            'model_paths': {'DL': {'model': config.bundle_root / 'models/dl.keras'},
                            'ML': {'p0': config.bundle_root / 'models/model_p0.joblib', 'p1': config.bundle_root / 'models/model_p1.joblib'}},
            'support': config.bundle_root / 'support'}


def replay_fingerprint(config):
    paths = artifact_paths(config)
    required = ['evidence/freeze_manifest.json', 'evidence/preprocessing_manifest.json', 'evidence/test_input_manifest.json',
                'evidence/results.json', 'evidence/summary_manifest.json', 'inputs/customers.npz',
                'predictions/final_predictions.npz', 'summaries/summaries.npz',
                *['aggregates/' + n for n in FINAL_AGGREGATE_SHA256]]
    files = {relative: sha256(_checked(config, relative)) for relative in required}
    files.update({'deployment_source:' + n: sha256(CODE_DIR / n) for n in ('artifacts.py', 'selection.py', 'streamlit_app.py')})
    payload = {'files': files, 'deployment': config.payload(), 'experiment_commit': EXPERIMENT_COMMIT, 'freeze_sha256': FREEZE_SHA256}
    return {**payload, 'digest': canonical_hash(payload)}


@dataclass
class ReplayData:
    ids: np.ndarray
    static: np.ndarray
    predictions: dict
    summaries: dict
    meta: dict


def _validate_input(config, p):
    freeze, manifest = p['freeze'], p['bundle']
    train = read_json(_checked(config, 'evidence/preprocessing_manifest.json'))
    require(train['train_customers'] == 120023 and train['final_test_included'] is False, 'Unexpected preprocessing fit population')
    names = train['static_features']
    require(len(names) == 91 and names == p['input_meta']['static_features'] == freeze['preprocessing']['static_features'], 'Frozen 91-variable names/order mismatch')
    feature_hash = hashlib.sha256(json.dumps(names, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    require(feature_hash == freeze['ml']['settings']['feature_order_sha256'] == manifest['feature_order_sha256'], 'Frozen variable order hash mismatch')
    with np.load(_checked(config, 'inputs/customers.npz'), allow_pickle=False) as data:
        require(set(data.files) == {'client_id', 'static'}, 'Deployment inputs must not contain outcomes or assignment')
        ids, static = data['client_id'], data['static']
    require(ids.shape == (40008,) and static.shape == (40008, 91) and static.dtype == np.float32, 'Fixed cohort/input shape mismatch')
    require(len(np.unique(ids)) == 40008 and np.isfinite(static).all(), 'Invalid fixed deployment input')
    order_hash = hashlib.sha256(ids.tobytes()).hexdigest()
    require(order_hash == p['input_meta']['customer_order_sha256'] == manifest['customer_order_sha256'], 'Fixed internal customer ID/order mismatch')
    require(hashlib.sha256(static.tobytes()).hexdigest() == manifest['static_array_sha256'], 'Static values differ from the finalized input')
    require(p['input_meta']['preprocessing_refit'] is False and p['input_meta']['training_state_reused_without_refit'] is True, 'Preprocessing refit is disallowed')
    return ids, static, names, feature_hash


def load_replay(config):
    p = artifact_paths(config)
    ids, static, names, feature_hash = _validate_input(config, p)
    with np.load(_checked(config, 'predictions/final_predictions.npz'), allow_pickle=False) as data:
        require(set(data.files) == {'client_id', 'dl_p0', 'dl_p1', 'ml_p0', 'ml_p1'}, 'Unexpected prediction columns')
        require(np.array_equal(ids, data['client_id']), 'Stored prediction customer order mismatch')
        predictions = {kind: np.column_stack([data[prefix + '_p0'], data[prefix + '_p1']]) for kind, prefix in [('DL', 'dl'), ('ML', 'ml')]}
        for key in ('dl_p0', 'dl_p1', 'ml_p0', 'ml_p1'):
            require(hashlib.sha256(data[key].tobytes()).hexdigest() == p['bundle']['prediction_array_sha256'][key], 'Stored probability values differ from source')
    from selection import select_indices
    for values in predictions.values():
        select_indices(ids, values)
    summary_meta = read_json(_checked(config, 'evidence/summary_manifest.json'))
    require(summary_meta['status'] == 'passed' and summary_meta['frozen_retransform_exact_match'], 'Original-unit summary audit did not pass')
    require(summary_meta['freeze_sha256'] == FREEZE_SHA256 and summary_meta['preprocessing_sha256'] == p['freeze']['preprocessing']['manifest_sha256'], 'Original-unit summary version mismatch')
    with np.load(_checked(config, 'summaries/summaries.npz'), allow_pickle=False) as data:
        require(set(data.files) == {'client_id', *SUMMARY_FIELDS} and np.array_equal(ids, data['client_id']), 'Original-unit customer summaries mismatch')
        summaries = {name: data[name] for name in SUMMARY_FIELDS}
    require(all(values.shape == (40008,) and np.isfinite(values).all() for values in summaries.values()), 'Invalid original-unit summaries')
    for a in [ids, static, *predictions.values(), *summaries.values()]:
        a.flags.writeable = False
    return ReplayData(ids, static, predictions, summaries, {'experiment_commit': EXPERIMENT_COMMIT, 'freeze_sha256': FREEZE_SHA256,
                      'input_version': 'reduced-v1/protected-portable-v1', 'feature_order_sha256': feature_hash,
                      'customer_order_sha256': p['input_meta']['customer_order_sha256'], 'static_features': names,
                      'reference': '2019-03-19 00:00:00', 'summary_status': '원자료 영수증 단위 요약 · 최종 전처리 재변환 일치 확인',
                      'prediction_sha256': p['bundle']['files']['predictions/final_predictions.npz']['sha256']})


def load_aggregates(config):
    p = artifact_paths(config)
    for name, expected in FINAL_AGGREGATE_SHA256.items():
        require(sha256(_checked(config, 'aggregates/' + name)) == expected, 'Final aggregate version mismatch')
    bootstrap = read_json(p['output_dir'] / 'bootstrap_results.json')
    require(bootstrap['status'] == 'passed' and bootstrap['freeze_sha256'] == FREEZE_SHA256 and bootstrap['valid_replicates'] == 2000, 'Final aggregate record mismatch')
    with (p['output_dir'] / 'test_scorecard.csv').open(encoding='utf-8-sig', newline='') as stream:
        scorecard = list(csv.DictReader(stream))
    with (p['output_dir'] / 'bootstrap_intervals.csv').open(encoding='utf-8-sig', newline='') as stream:
        intervals = list(csv.DictReader(stream))
    return {'results': p['results'], 'bootstrap': bootstrap, 'scorecard': scorecard, 'intervals': intervals}


def environment_metadata():
    packages = {}
    for name in ('numpy', 'pandas', 'scipy', 'tensorflow', 'tensorflow-cpu', 'keras', 'h5py', 'protobuf', 'scikit-learn', 'joblib', 'threadpoolctl', 'streamlit'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    return {'python': platform.python_version(), 'implementation': platform.python_implementation(), 'platform': platform.platform(),
            'machine': platform.machine(), 'packages': packages,
            'thread_policy': {'tensorflow_intra': 2, 'tensorflow_inter': 1, 'ml_threadpool': 2},
            'environment': {name: os.environ.get(name) for name in ('TF_ENABLE_ONEDNN_OPTS', 'TF_DETERMINISTIC_OPS', 'CUDA_VISIBLE_DEVICES', 'OMP_NUM_THREADS')}}


def inference_fingerprint(config, model):
    require(model in ('DL', 'ML'), 'Unknown model')
    p = artifact_paths(config)
    ids, static, names, feature_hash = _validate_input(config, p)
    required = ['evidence/freeze_manifest.json', 'evidence/preprocessing_manifest.json', 'evidence/test_input_manifest.json',
                'evidence/results.json', 'inputs/customers.npz', 'predictions/final_predictions.npz',
                'support/model_objects.py', 'support/runtime.py']
    model_files = ['models/dl.keras'] if model == 'DL' else ['models/model_p0.joblib', 'models/model_p1.joblib']
    files = {name: sha256(_checked(config, name)) for name in required + model_files}
    for name in ('model_objects.py', 'runtime.py'):
        require(files['support/' + name] == p['freeze']['code_version']['files_sha256']['support\\' + name],
                'Original frozen inference support source differs')
    expected = {'models/dl.keras': p['freeze']['dl']['checkpoint_sha256'], **{'models/model_' + head + '.joblib': value for head, value in p['freeze']['ml']['model_sha256'].items()}}
    for name in model_files:
        require(files[name] == expected[name], 'Frozen model content differs')
    files.update({'deployment_source:' + name: sha256(CODE_DIR / name) for name in INFERENCE_CODE})
    files.update({'dependency_spec:' + path.name: sha256(path) for path in sorted(CODE_DIR.glob('requirements*.txt'))})
    payload = {'model': model, 'files': files, 'deployment': config.payload(), 'environment': environment_metadata(),
               'experiment_commit': EXPERIMENT_COMMIT, 'freeze_sha256': FREEZE_SHA256, 'feature_order_sha256': feature_hash,
               'customer_order_sha256': p['input_meta']['customer_order_sha256'], 'atol': ATOL, 'rtol': RTOL}
    return {**payload, 'digest': canonical_hash(payload)}


def inference_status(config, model):
    try:
        fingerprint = inference_fingerprint(config, model)
        pointer_path = config.record_dir / (model + '.latest.json')
        if pointer_path.is_file():
            # A local failed/corrupt event always overrides an older saved PASS.
            pointer = read_json(pointer_path)
            record_path = config.record_dir / pointer['record_file']
            require(record_path.resolve().parent == config.record_dir.resolve(), 'Invalid verification record path')
            require(sha256(record_path) == pointer['record_sha256'], 'Verification record changed')
            record = read_json(record_path)
        else:
            require('streamlit' in sys.modules, 'No server-context verification record')
            import streamlit as st
            saved = json.loads(str(st.secrets['deployment_verification']['records_json']))
            records = saved['records'] if isinstance(saved, dict) else saved
            matches = [item for item in records if item.get('model') == model]
            require(len(matches) == 1, 'Missing or ambiguous persisted server verification record')
            record = matches[0]
        require(record['model'] == model and record['status'] == 'passed', 'Deployment environment check has not passed')
        require(record['fingerprint']['digest'] == fingerprint['digest'], 'Current deployment artifacts/code/environment do not match the PASS record')
        require(record['atol'] == ATOL and record['rtol'] == RTOL and record['probabilities_match'] and record['selected_set_exact_match'], 'Invalid acceptance criteria')
        require(record['selected_symmetric_difference'] == 0 and record['customer_count'] == 40008, 'Incomplete reproduction record')
        return {'enabled': True, 'reason': '현재 배포 환경·파일·코드의 전체 고객 재현 검사 통과', 'record': record, 'fingerprint': fingerprint}
    except Exception:
        # Do not expose host paths, secret keys, or protected customer values.
        return {'enabled': False, 'reason': '현재 배포 환경과 일치하는 재현 검사 통과 기록이 필요합니다.', 'record': None, 'fingerprint': None}


def disable_inference(config, model, failure_type='RuntimeError'):
    from datetime import datetime, timezone
    require(model in ('DL', 'ML'), 'Unknown model')
    now = datetime.now(timezone.utc)
    record = {'model': model, 'status': 'failed', 'checked_at_utc': now.isoformat(), 'atol': ATOL, 'rtol': RTOL,
              'experiment_commit': EXPERIMENT_COMMIT, 'freeze_sha256': FREEZE_SHA256,
              'failure_type': failure_type, 'failure_reason': 'Live inference failed; a new separate deployment-environment check is required.',
              'probabilities_match': False, 'selected_set_exact_match': False}
    config.record_dir.mkdir(parents=True, exist_ok=True)
    name = model + '.runtime_failed.' + now.strftime('%Y%m%dT%H%M%S%f') + '.json'
    path = config.record_dir / name
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    (config.record_dir / (model + '.latest.json')).write_text(json.dumps({'record_file': name, 'record_sha256': sha256(path)}), encoding='utf-8')
