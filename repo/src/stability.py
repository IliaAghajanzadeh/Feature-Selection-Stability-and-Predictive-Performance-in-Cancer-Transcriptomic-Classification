"""
Nogueira, Sechidis & Brown (JMLR 2018) stability estimator, plus two
explicit ways to turn "one Pareto front per bootstrap" into "one binary
mask per bootstrap" (which is what the Nogueira estimator assumes as
input). This ambiguity was flagged before writing any code -- both
definitions are computed and logged; which one goes in the paper is a
decision for after we see real numbers, not before.

Nogueira formula (closed-form, O(M*p), avoids pairwise Jaccard):
  Given Z: (M runs) x (p features) binary matrix.
  k_i      = features selected in run i (row sum)
  kbar     = mean(k_i)
  phat_f   = mean over runs of Z[:,f]           (selection frequency of feature f)
  numer    = (1/p) * sum_f phat_f*(1-phat_f) * M/(M-1)      [unbiased per-feature variance]
  denom    = (kbar/p) * (1 - kbar/p)                         [variance under a null model]
  stability = 1 - numer/denom                                [ -> 1 = perfectly stable ]
Undefined (denom=0) when kbar=0 or kbar=p for every run; returns np.nan in that
degenerate case rather than dividing by zero.
"""
import numpy as np


def nogueira_stability(Z):
    """Z: array-like, shape (M, p), binary. Returns (stability, detail dict)."""
    Z = np.asarray(Z, dtype=float)
    M, p = Z.shape
    if M < 2:
        raise ValueError("Nogueira stability needs at least 2 runs (bootstraps).")
    k = Z.sum(axis=1)
    kbar = k.mean()
    phat = Z.mean(axis=0)
    numer = (phat * (1 - phat)).mean() * (M / (M - 1))
    denom = (kbar / p) * (1 - kbar / p)
    if denom <= 1e-12:
        return float("nan"), dict(M=M, p=p, kbar=kbar, numer=numer, denom=denom, note="degenerate: kbar is 0 or p for all runs")
    stab = 1.0 - numer / denom
    return float(stab), dict(M=M, p=p, kbar=float(kbar), numer=float(numer), denom=float(denom))


def masks_to_matrix(masks, n_features):
    """List of boolean arrays (possibly ragged in the sense of coming from
    different individuals but all length n_features) -> (M, p) matrix."""
    Z = np.zeros((len(masks), n_features), dtype=int)
    for i, m in enumerate(masks):
        Z[i, :] = np.asarray(m, dtype=int)
    return Z


def front_level_mask(front, n_features):
    """DEFINITION: a feature counts as 'selected by this bootstrap' if it
    appears in AT LEAST ONE Pareto-optimal subset from that bootstrap's run.
    front: list of (mask, acc, n_sel) tuples as returned by ga.run_nsga2.
    This is a union, not a weighted vote -- documented explicitly because
    it is a real modeling choice, not the only reasonable one."""
    if len(front) == 0:
        return np.zeros(n_features, dtype=int)
    union = np.zeros(n_features, dtype=bool)
    for mask, acc, n_sel in front:
        union |= mask
    return union.astype(int)


def knee_point_mask(front):
    """DEFINITION: point-level representative = knee of the Pareto front,
    found by normalizing both objectives to [0,1] over THIS front and
    minimizing distance to the ideal point (acc=1, n_sel=0).
    front: list of (mask, acc, n_sel)."""
    if len(front) == 0:
        return None
    if len(front) == 1:
        return front[0][0]
    accs = np.array([a for _, a, _ in front])
    nsels = np.array([n for _, _, n in front])
    a_rng = accs.max() - accs.min()
    n_rng = nsels.max() - nsels.min()
    acc_norm = (accs - accs.min()) / a_rng if a_rng > 0 else np.zeros_like(accs)
    nsel_norm = (nsels - nsels.min()) / n_rng if n_rng > 0 else np.zeros_like(nsels)
    dist = np.sqrt((1 - acc_norm) ** 2 + nsel_norm ** 2)
    return front[int(np.argmin(dist))][0]


def compute_both_stabilities(fronts, n_features):
    """fronts: list (length = #bootstraps) of Pareto fronts (each a list of
    (mask, acc, n_sel)). Returns dict with both operationalizations."""
    front_masks = [front_level_mask(f, n_features) for f in fronts]
    point_masks = [knee_point_mask(f) for f in fronts]
    point_masks = [m for m in point_masks if m is not None]

    Z_front = masks_to_matrix(front_masks, n_features)
    Z_point = masks_to_matrix(point_masks, n_features)

    stab_front, detail_front = nogueira_stability(Z_front)
    stab_point, detail_point = nogueira_stability(Z_point)
    return {
        "front_level": {"stability": stab_front, **detail_front},
        "point_level": {"stability": stab_point, **detail_point},
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p = 50
    # perfectly stable case: same 5 features every run
    Z_stable = np.zeros((10, p), dtype=int); Z_stable[:, :5] = 1
    print("identical masks every run ->", nogueira_stability(Z_stable))

    # unstable case: random 5-of-50 each run
    Z_unstable = np.zeros((10, p), dtype=int)
    for i in range(10):
        idx = rng.choice(p, 5, replace=False)
        Z_unstable[i, idx] = 1
    print("random 5-of-50 each run    ->", nogueira_stability(Z_unstable))
