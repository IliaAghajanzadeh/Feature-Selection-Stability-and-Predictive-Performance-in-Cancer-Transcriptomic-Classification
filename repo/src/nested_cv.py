"""
LEAKAGE-CRITICAL MODULE. Read this file first when auditing the pipeline.

TWO distinct leakage risks are guarded here, not one:

(1) Scaling/preprocessing leakage: the scaler is always .fit() on the
    inner-training fold only, then .transform() on both train and
    validation folds. No statistic computed from validation data ever
    touches training.

(2) Duplicate-sample leakage from bootstrap resampling: when X,y come from
    a WITH-REPLACEMENT bootstrap draw (see bootstrap.py), the same original
    sample can appear multiple times as different rows. A naive CV split
    on row position can then put one copy of a patient in the training
    fold and another copy of THE SAME patient in the validation fold --
    the model has effectively already seen the "held-out" case. This is
    silent and does not raise any error; it just inflates accuracy and can
    distort which feature subsets look best. Fixed here via GroupKFold /
    LeaveOneGroupOut using the ORIGINAL (pre-resampling) sample index as
    the group label, so every duplicate of a given original sample is
    forced to the same side of every split. Empirically, about 40% of rows
    in a size-72 bootstrap are duplicates of an already-included sample
    (see audit note in conversation) -- this is not a rare edge case.

The feature mask is chosen by the GA using ONLY the fitness values this
module returns.
"""
import numpy as np
from sklearn.model_selection import StratifiedKFold, LeaveOneOut, LeaveOneGroupOut, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.exceptions import ConvergenceWarning
import warnings

warnings.filterwarnings("ignore", category=ConvergenceWarning)


def make_cv(n_samples, n_splits=5, loo_threshold=100, seed=0, groups=None):
    """LOOCV for very small n, else stratified/grouped k-fold.
    groups: array of length n_samples giving the ORIGINAL sample identity
    for each row (needed when the data has been bootstrap-resampled with
    replacement, so duplicate rows of the same original sample never get
    split across train/validation -- see module docstring update below).
    If groups is None, every row is assumed to be its own unique sample
    (correct for un-resampled data)."""
    if groups is not None and len(np.unique(groups)) < n_samples:
        # duplicates present -> must use group-aware splitting
        n_unique = len(np.unique(groups))
        if n_unique <= loo_threshold:
            return LeaveOneGroupOut()
        return GroupKFold(n_splits=n_splits)
    if n_samples <= loo_threshold:
        return LeaveOneOut()
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)


def eval_mask(X, y, mask, cv, classifier_fn=None, groups=None):
    """Inner-CV balanced accuracy for one feature mask on one dataset.
    groups: passed straight to cv.split(); required (and must match what
    make_cv() was built with) whenever X,y come from a bootstrap resample
    with duplicate rows -- otherwise duplicate copies of the same original
    sample can straddle the train/validation boundary."""
    if mask.sum() == 0:
        return 0.0
    if classifier_fn is None:
        classifier_fn = lambda: KNeighborsClassifier(n_neighbors=5)

    Xs = X[:, mask]
    accs = []
    for tr_idx, va_idx in cv.split(Xs, y, groups=groups):
        Xtr, Xva = Xs[tr_idx], Xs[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        scaler = StandardScaler()
        Xtr = scaler.fit_transform(Xtr)      # fit ONLY on training fold
        Xva = scaler.transform(Xva)          # validation fold only transformed

        if len(np.unique(ytr)) < 2:
            # degenerate fold (can happen with LOOCV): fall back to majority class
            pred = np.full(len(yva), ytr[0])
        else:
            clf = classifier_fn()
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xva)

        accs.append(_balanced_acc(yva, pred))
    return float(np.mean(accs))


def _balanced_acc(y_true, y_pred):
    """Balanced accuracy, computed by hand (avoids sklearn's warning noise
    on single-sample LOOCV folds) -- macro-average of per-class recall over
    classes actually present in y_true for this fold."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    classes = np.unique(y_true)
    if len(classes) == 1:
        return float((y_pred == y_true).mean())
    recalls = []
    for c in classes:
        m = y_true == c
        recalls.append((y_pred[m] == c).mean())
    return float(np.mean(recalls))


if __name__ == "__main__":
    # smoke test on a trivial synthetic case
    rng = np.random.default_rng(0)
    n, p = 60, 20
    X = rng.normal(size=(n, p))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)  # only features 0,1 matter
    cv = make_cv(n)
    mask_good = np.zeros(p, dtype=bool); mask_good[[0, 1]] = True
    mask_bad = np.zeros(p, dtype=bool); mask_bad[[5, 6]] = True
    print("informative mask balanced-acc:", eval_mask(X, y, mask_good, cv))
    print("random mask balanced-acc     :", eval_mask(X, y, mask_bad, cv))
