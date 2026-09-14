"""
Outer loop: draw B bootstrap resamples of the dataset, run one independent
NSGA-II (with its own inner CV) per bootstrap, collect Pareto fronts,
compute both stability definitions.

Each bootstrap resample is drawn WITH replacement, same size as original
(standard .632 bootstrap for this purpose is a later refinement, not needed
for a kill test). Stratified by class to avoid a resample that accidentally
loses one class entirely on small datasets.
"""
import time
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

import data as data_mod
import nested_cv as ncv
import ga as ga_mod
import stability as stab_mod


def stratified_bootstrap_indices(y, rng):
    """With-replacement resample, but drawn separately within each class so
    the class ratio of the resample matches the original (avoids a bootstrap
    that wipes out the minority class on n=62-72 datasets)."""
    idx = np.arange(len(y))
    out = []
    for c in np.unique(y):
        c_idx = idx[y == c]
        out.append(rng.choice(c_idx, size=len(c_idx), replace=True))
    return np.concatenate(out)


def run_bootstrap_study(dataset_name, n_bootstraps, seed=0,
                         pop_size=60, n_gen=40, verbose=True):
    X, y, tag = data_mod.load(dataset_name)
    n, p = X.shape
    rng = np.random.default_rng(seed)

    fronts = []
    t0 = time.time()
    for b in range(n_bootstraps):
        bidx = stratified_bootstrap_indices(y, rng)
        Xb, yb = X[bidx], y[bidx]
        groups = bidx  # original-sample identity: duplicates share this label

        cv = ncv.make_cv(len(yb), seed=seed + b, groups=groups)
        eval_fn = lambda mask, Xb=Xb, yb=yb, cv=cv, groups=groups: ncv.eval_mask(Xb, yb, mask, cv, groups=groups)

        front = ga_mod.run_nsga2(p, eval_fn, pop_size=pop_size, n_gen=n_gen, seed=seed + b)
        front = [(m, a, k) for (m, a, k) in front if k > 0]  # drop degenerate empty-mask point
        fronts.append(front)

        if verbose:
            best = max(front, key=lambda t: t[1]) if front else (None, 0, 0)
            print(f"[{tag}] bootstrap {b+1}/{n_bootstraps}  "
                  f"front_size={len(front)}  best_acc={best[1]:.3f} (k={best[2]})  "
                  f"[{time.time()-t0:.0f}s]")

    result = stab_mod.compute_both_stabilities(fronts, p)
    result["dataset"] = tag
    result["n_samples"] = n
    result["n_features"] = p
    result["n_bootstraps"] = n_bootstraps
    result["fronts"] = fronts  # kept for later inspection / plotting
    return result


def run_seed_control_study(dataset_name, n_seeds, base_seed=0,
                            pop_size=40, n_gen=20, verbose=True):
    """CONTROL EXPERIMENT: same, un-resampled data every run; only the
    NSGA-II random seed changes. Isolates optimizer stochasticity from the
    data-resampling instability we're actually trying to measure. No
    groups/duplicates issue here since there is no bootstrap draw."""
    X, y, tag = data_mod.load(dataset_name)
    n, p = X.shape
    cv = ncv.make_cv(n)  # built once; same split logic reused, but note
                          # StratifiedKFold(shuffle=True) still depends on
                          # its own random_state, fixed here to base_seed
                          # so fold composition itself does not vary across
                          # runs -- only the GA's search randomness does.

    fronts = []
    t0 = time.time()
    for s in range(n_seeds):
        eval_fn = lambda mask, X=X, y=y, cv=cv: ncv.eval_mask(X, y, mask, cv)
        front = ga_mod.run_nsga2(p, eval_fn, pop_size=pop_size, n_gen=n_gen, seed=base_seed + s)
        front = [(m, a, k) for (m, a, k) in front if k > 0]
        fronts.append(front)
        if verbose:
            best = max(front, key=lambda t: t[1]) if front else (None, 0, 0)
            print(f"[{tag} SEED-CONTROL] seed {s+1}/{n_seeds}  "
                  f"front_size={len(front)}  best_acc={best[1]:.3f} (k={best[2]})  "
                  f"[{time.time()-t0:.0f}s]")

    result = stab_mod.compute_both_stabilities(fronts, p)
    result["dataset"] = tag + "_SEED_CONTROL"
    result["n_samples"] = n
    result["n_features"] = p
    result["n_bootstraps"] = n_seeds
    result["fronts"] = fronts
    return result


if __name__ == "__main__":
    import sys
    dataset = sys.argv[1] if len(sys.argv) > 1 else "breast_cancer"
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    res = run_bootstrap_study(dataset, n_bootstraps=B, seed=0, pop_size=60, n_gen=30)
    print("\n=== RESULT ===")
    print(f"dataset={res['dataset']} n={res['n_samples']} p={res['n_features']} B={res['n_bootstraps']}")
    print(f"front-level stability: {res['front_level']['stability']:.4f}  detail={res['front_level']}")
    print(f"point-level stability: {res['point_level']['stability']:.4f}  detail={res['point_level']}")


def run_seed_control_matched(dataset_name, n_seeds, boot_seed=0, base_seed=100,
                              pop_size=40, n_gen=20, verbose=True):
    """MATCHED seed-only control (supervisor-review fix).

    The original run_seed_control_study() held the ORIGINAL dataset fixed,
    which meant it used a different inner-CV scheme from the bootstrap
    condition (LeaveOneOut on 72 unique samples vs LeaveOneGroupOut on ~44
    groups). That confounded the bootstrap-vs-seed comparison with a
    difference in fold count and effective training size.

    Here we instead hold ONE bootstrap draw fixed across all runs. Both
    conditions then have identical row counts, identical duplicate
    structure, and identical CV folds; the only remaining difference is
    whether the draw is redrawn between runs. This is the correct control.
    """
    X, y, tag = data_mod.load(dataset_name)
    n, p = X.shape
    rng = np.random.default_rng(boot_seed)
    bidx = stratified_bootstrap_indices(y, rng)     # ONE draw, reused every run
    Xb, yb = X[bidx], y[bidx]
    groups = bidx
    cv = ncv.make_cv(len(yb), seed=boot_seed, groups=groups)

    fronts = []
    t0 = time.time()
    for s in range(n_seeds):
        eval_fn = lambda mask, Xb=Xb, yb=yb, cv=cv, g=groups: ncv.eval_mask(Xb, yb, mask, cv, groups=g)
        front = ga_mod.run_nsga2(p, eval_fn, pop_size=pop_size, n_gen=n_gen, seed=base_seed + s)
        front = [(m, a, k) for (m, a, k) in front if k > 0]
        fronts.append(front)
        if verbose:
            best = max(front, key=lambda t: t[1]) if front else (None, 0, 0)
            print(f"[{tag} SEED-MATCHED] seed {s+1}/{n_seeds}  front={len(front)}  "
                  f"best_acc={best[1]:.3f} (k={best[2]})  [{time.time()-t0:.0f}s]", flush=True)

    result = stab_mod.compute_both_stabilities(fronts, p)
    result["dataset"] = tag + "_SEED_MATCHED"
    result["n_samples"] = n
    result["n_features"] = p
    result["n_bootstraps"] = n_seeds
    result["cv_scheme"] = type(cv).__name__
    result["n_folds"] = int(cv.get_n_splits(Xb, yb, groups=groups))
    result["fronts"] = fronts
    return result
