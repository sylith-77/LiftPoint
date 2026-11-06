"""Frozen ranking policy. Never consumes treatment assignment or outcomes."""
import hashlib
import numpy as np

CUSTOMERS = 40008
SELECTED = 12002
TIE_PREFIX = 'IncrementalValue-tie-v1:'


def select_indices(ids, probabilities):
    """Return all indices ranked by unrounded p1-p0, original SHA, internal ID."""
    ids = np.asarray(ids)
    p = np.asarray(probabilities)
    if ids.shape != (CUSTOMERS,) or p.shape != (CUSTOMERS, 2):
        raise ValueError('Expected the fixed 40,008-customer cohort and two output columns')
    if len(np.unique(ids)) != CUSTOMERS or not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Invalid identifiers or probability array; customer values suppressed')
    # Preserve the saved dtype when subtracting, exactly as the final evaluation.
    uplift = p[:, 1] - p[:, 0]
    tie = np.asarray([hashlib.sha256((TIE_PREFIX + str(cid)).encode()).hexdigest() for cid in ids])
    return np.lexsort((ids.astype(str), tie, -uplift))


def anonymous_ids(ids):
    values = np.asarray(['LP-' + hashlib.sha256(('LiftPoint-display-v1:' + str(cid)).encode()).hexdigest()[:16].upper() for cid in ids])
    if len(np.unique(values)) != len(values):
        raise ValueError('Anonymous display identifier collision')
    return values
