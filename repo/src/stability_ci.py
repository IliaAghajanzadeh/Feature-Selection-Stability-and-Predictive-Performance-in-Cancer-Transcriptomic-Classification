"""
Confidence interval for the Nogueira et al. (JMLR 2018) stability estimator.

Two independent methods, deliberately both implemented so they can be
cross-checked against each other (if an asymptotic CI and a bootstrap CI
disagree badly at M=10, that itself is the finding -- it means M is too
small for the asymptotic approximation and we must report the bootstrap one).

METHOD A -- asymptotic (Nogueira et al., Sec. 5).
  Treats the M runs as iid draws. The estimator is
      stab = 1 - (mean_f s_f^2) / (kbar/p * (1 - kbar/p))
  where s_f^2 = M/(M-1) * phat_f(1-phat_f) is the unbiased per-feature
  variance. The variance of stab is obtained by the delta method over the
  per-run contributions; we compute it empirically via the influence-function
  form, which is what makes it valid without assuming a distribution on Z.

METHOD B -- nonparametric bootstrap over RUNS.
  Resample the M runs (rows of Z) with replacement B times, recompute
  stability each time, take empirical percentiles. Makes no asymptotic
  assumption. With M=10 this is coarse (only 10 distinct rows to draw from)
  but it is honest about that coarseness rather than hiding it.

IMPORTANT LIMITATION, applies to both: with M=10 runs, ANY interval on a
stability coefficient will be wide. These CIs exist to be reported honestly,
not to manufacture significance.
"""
import numpy as np


def nogueira_stability(Z):
    """Point estimate. Z: (M, p) binary. Returns (stability, detail)."""
    Z = np.asarray(Z, dtype=float)
    M, p = Z.shape
    if M < 2:
        raise ValueError("need >= 2 runs")
    k = Z.sum(axis=1)
    kbar = k.mean()
    phat = Z.mean(axis=0)
    numer = (phat * (1 - phat)).mean() * (M / (M - 1))
    denom = (kbar / p) * (1 - kbar / p)
    if denom <= 1e-12:
        return float("nan"), dict(M=M, p=p, kbar=kbar, note="degenerate")
    return float(1.0 - numer / denom), dict(M=M, p=p, kbar=float(kbar))


def _stability_from_rows(Z):
    s, _ = nogueira_stability(Z)
    return s


def ci_asymptotic(Z, alpha=0.05):
    """METHOD A. Influence-function / jackknife-style asymptotic interval.
    We use the jackknife over runs to estimate Var(stab), which is a
    consistent estimator of the asymptotic variance and avoids having to
    hand-differentiate the ratio."""
    Z = np.asarray(Z, dtype=float)
    M = Z.shape[0]
    full = _stability_from_rows(Z)
    if not np.isfinite(full):
        return (float("nan"), float("nan"), float("nan"))

    # leave-one-run-out replicates
    reps = []
    for i in range(M):
        Zi = np.delete(Z, i, axis=0)
        s = _stability_from_rows(Zi)
        if np.isfinite(s):
            reps.append(s)
    reps = np.array(reps)
    if len(reps) < 2:
        return (full, float("nan"), float("nan"))

    m = len(reps)
    var_jack = (m - 1) / m * np.sum((reps - reps.mean()) ** 2)
    se = float(np.sqrt(max(var_jack, 0.0)))
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    return (float(full), float(full - z * se), float(full + z * se))


def ci_bootstrap(Z, alpha=0.05, B=5000, seed=0):
    """METHOD B. Nonparametric bootstrap over runs (rows)."""
    Z = np.asarray(Z, dtype=float)
    M = Z.shape[0]
    rng = np.random.default_rng(seed)
    full = _stability_from_rows(Z)
    vals = []
    for _ in range(B):
        idx = rng.integers(0, M, M)
        Zb = Z[idx]
        s = _stability_from_rows(Zb)
        if np.isfinite(s):
            vals.append(s)
    if len(vals) < 100:
        return (float(full), float("nan"), float("nan"))
    vals = np.array(vals)
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return (float(full), lo, hi)


def ci_both(Z, alpha=0.05, seed=0):
    est, a_lo, a_hi = ci_asymptotic(Z, alpha)
    _, b_lo, b_hi = ci_bootstrap(Z, alpha, seed=seed)
    return {
        "stability": est,
        "ci_asymptotic": [a_lo, a_hi],
        "ci_bootstrap": [b_lo, b_hi],
        "alpha": alpha,
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p, M = 2000, 10

    print("--- VALIDATION 1: perfectly stable (same 10 features every run) ---")
    Z = np.zeros((M, p), dtype=int); Z[:, :10] = 1
    print("  expect stability=1, CI degenerate at 1:", ci_both(Z))

    print("\n--- VALIDATION 2: fully random (10-of-2000 independently each run) ---")
    Z = np.zeros((M, p), dtype=int)
    for i in range(M):
        Z[i, rng.choice(p, 10, replace=False)] = 1
    r = ci_both(Z)
    print(f"  expect stability~0 and CI straddling 0: {r}")
    lo_a, hi_a = r["ci_asymptotic"]
    print(f"  CI includes 0? asymptotic={lo_a <= 0 <= hi_a}")

    print("\n--- VALIDATION 3: partially stable (5 fixed + 5 random each run) ---")
    Z = np.zeros((M, p), dtype=int)
    for i in range(M):
        Z[i, :5] = 1
        Z[i, rng.choice(np.arange(5, p), 5, replace=False)] = 1
    print("  expect intermediate stability:", ci_both(Z))

    print("\n--- VALIDATION 4: does CI width shrink as M grows? (should) ---")
    for M2 in [5, 10, 30, 100]:
        Z = np.zeros((M2, p), dtype=int)
        for i in range(M2):
            Z[i, :5] = 1
            Z[i, rng.choice(np.arange(5, p), 5, replace=False)] = 1
        r = ci_both(Z)
        lo, hi = r["ci_asymptotic"]
        print(f"  M={M2:>4}: stab={r['stability']:.4f}  asym CI width={hi-lo:.4f}")


# =============================================================================
# DESCRIPTIVE ALTERNATIVE (recommended over the CIs above at M=10)
# =============================================================================
def pairwise_jaccard(Z):
    """Distribution of pairwise Jaccard similarity between runs.
    Unlike the CIs above this needs NO asymptotic assumption and no
    resampling, so it is valid at M=10. Report median + IQR + min/max.
    Interpretation is direct: 'two independent runs shared X% of their
    selected features'."""
    Z = np.asarray(Z, dtype=bool)
    M = Z.shape[0]
    vals = []
    for i in range(M):
        for j in range(i + 1, M):
            inter = np.logical_and(Z[i], Z[j]).sum()
            union = np.logical_or(Z[i], Z[j]).sum()
            if union > 0:
                vals.append(inter / union)
    vals = np.array(vals)
    return {
        "n_pairs": len(vals),
        "median": float(np.median(vals)),
        "q1": float(np.percentile(vals, 25)),
        "q3": float(np.percentile(vals, 75)),
        "min": float(vals.min()),
        "max": float(vals.max()),
    }
