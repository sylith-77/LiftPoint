"""Lazy fixed-model inference. Importing this module does not import any ML library."""
from functools import lru_cache
import importlib.util
from pathlib import Path
import numpy as np
from artifacts import artifact_paths, disable_inference, inference_fingerprint, inference_status, require, _validate_input


def _source_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@lru_cache(maxsize=2)
def _dl_model(path, content_key, support):
    import tensorflow as tf
    try:
        tf.config.threading.set_intra_op_parallelism_threads(2)
        tf.config.threading.set_inter_op_parallelism_threads(1)
    except RuntimeError:
        # The app can reuse a loaded TensorFlow runtime only with the same settings.
        require(tf.config.threading.get_intra_op_parallelism_threads() == 2 and tf.config.threading.get_inter_op_parallelism_threads() == 1,
                'TensorFlow threading differs from the fixed inference environment')
    objects = _source_module('liftpoint_frozen_model_objects', Path(support) / 'model_objects.py')
    # Exactly the registered-name alias used in the successful final notebook 46.
    tf.keras.utils.get_custom_objects()['IncrementalValue>factual_bce'] = objects.factual_bce
    model = objects.load_checkpoint(path)
    require(tuple(model.input_shape) == (None, 91) and tuple(model.output_shape) == (None, 2), 'Frozen DL shape mismatch')
    output_inputs = model.get_layer('probabilities').input
    heads = [tensor._keras_history.operation.name for tensor in output_inputs]
    require(heads == ['p0', 'p1'], 'Frozen DL p0/p1 output order mismatch')
    runtime = _source_module('liftpoint_frozen_runtime', Path(support) / 'runtime.py')
    return model, runtime


@lru_cache(maxsize=4)
def _ml_model(path, content_key):
    import joblib
    return joblib.load(path)


def predict_frozen(config, model, *, _verification=False):
    """The standalone verifier alone can bypass the prior-pass activation gate."""
    if not _verification:
        gate = inference_status(config, model)
        require(gate['enabled'], '실제 모델 추론 비활성화: ' + gate['reason'])
    before = inference_fingerprint(config, model)
    paths = artifact_paths(config)
    ids, static, _, _ = _validate_input(config, paths)
    freeze = paths['freeze']
    if model == 'DL':
        checkpoint = str(paths['model_paths']['DL']['model'])
        loaded, runtime = _dl_model(checkpoint, before['digest'], str(paths['support']))

        class FixedInputs:
            indices = np.arange(len(ids), dtype=np.int64)
            batch_size = 64

            @staticmethod
            def inputs(indices):
                return {'customer': static[indices]}

        # Original bounded predictor explicitly uses model(inputs, training=False).
        weights_before = runtime.weights_sha256(loaded)
        prediction = runtime.predict_batches(loaded, FixedInputs())
        require(runtime.weights_sha256(loaded) == weights_before, 'DL weights changed during inference')
        details = {'output_order': ['p0', 'p1'], 'training': False, 'dropout_disabled': True,
                   'input_shape': [40008, 91], 'predict_batch_size': 64, 'weights_sha256': weights_before,
                   'loader': 'frozen support/model_objects.py load_checkpoint + notebook46 legacy alias',
                   'predictor': 'frozen support/runtime.py predict_batches'}
    elif model == 'ML':
        from threadpoolctl import threadpool_limits
        prediction = np.empty((len(ids), 2), dtype=np.float64)
        class_columns = {}
        with threadpool_limits(limits=2):
            for head, column in [('p0', 0), ('p1', 1)]:
                loaded = _ml_model(str(paths['model_paths']['ML'][head]), before['digest'])
                require(int(loaded.n_features_in_) == 91, 'Frozen ML input variable count mismatch')
                positive = np.flatnonzero(loaded.classes_ == 1)
                require(len(positive) == 1, 'ML purchase class 1 is missing or ambiguous')
                class_columns[head] = {'classes': loaded.classes_.tolist(), 'purchase_class_column': int(positive[0])}
                for start in range(0, len(ids), 4096):
                    block = static[start:start + 4096]
                    prediction[start:start + len(block), column] = loaded.predict_proba(block)[:, positive[0]]
        details = {'output_order': ['p0', 'p1'], 'purchase_class_columns': class_columns, 'input_shape': [40008, 91],
                   'predict_batch_size': 4096, 'predictor': 'final notebook46 predict_proba positive class 1 loop'}
    else:
        raise ValueError('Unknown model')
    require(prediction.shape == (40008, 2) and np.isfinite(prediction).all() and ((prediction >= 0) & (prediction <= 1)).all(),
            'Invalid inference output; probabilities suppressed')
    require(inference_fingerprint(config, model)['digest'] == before['digest'], 'Artifact/code/environment changed during inference')
    return prediction, details


def run_inference(config, model):
    try:
        prediction, _ = predict_frozen(config, model)
        return prediction
    except Exception as exc:
        disable_inference(config, model, type(exc).__name__)
        raise
