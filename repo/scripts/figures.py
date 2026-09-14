"""
Figures for: "Data Perturbation vs. Search Randomness: An Empirical Study of
Feature Selection Stability with NSGA-II"

Reads ONLY the saved raw results -- no re-running of any experiment.
Outputs 300-dpi PNG + PDF (vector, for camera-ready).
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from stability_ci import pairwise_jaccard, nogueira_stability

P = {"breast_cancer": 30, "colon": 2000, "leukemia": 7070}
NICE = {"breast_cancer": "Breast Cancer\n(p/n=0.05)",
        "colon": "Colon\n(p/n=32)",
        "leukemia": "Leukemia\n(p/n=98)"}
COND = {"bootstrap_fixed": "Data resampling", "seed_control": "Search randomness only (CV-matched)"}
C_BOOT, C_SEED = "#3b6ea5", "#c1666b"

plt.rcParams.update({
    "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
    "grid.alpha": 0.25, "grid.linestyle": "-", "axes.axisbelow": True,
})


def _front_of(run):
    """Breast Cancer's file stores each run as a bare list of Pareto points;
    the later Colon/Leukemia file wraps it as {"seed":..., "front":[...]}."""
    return run["front"] if isinstance(run, dict) else run


def masks_from_raw(raw_runs, p):
    front_masks, point_masks, accs, ks = [], [], [], []
    for run in raw_runs:
        front = _front_of(run)
        u = np.zeros(p, dtype=int)
        for sub in front:
            u[sub["features"]] = 1
        front_masks.append(u)
        if not front:
            continue
        a = np.array([s["acc"] for s in front]); k = np.array([s["n_sel"] for s in front])
        ar, kr = a.max() - a.min(), k.max() - k.min()
        an = (a - a.min()) / ar if ar > 0 else np.zeros_like(a)
        kn = (k - k.min()) / kr if kr > 0 else np.zeros_like(k)
        knee = front[int(np.argmin(np.sqrt((1 - an) ** 2 + kn ** 2)))]
        pm = np.zeros(p, dtype=int); pm[knee["features"]] = 1
        point_masks.append(pm)
        accs.append(knee["acc"]); ks.append(knee["n_sel"])
    return np.array(front_masks), np.array(point_masks), accs, ks


# ---- load ----
bc = json.load(open("breast_cancer_matched_result.json"))
fin = json.load(open("final_raw_results.json"))
mat = json.load(open("matched_seed_control.json"))   # CV-matched control (supersedes seed_control)
D = {}
D[("breast_cancer", "bootstrap_fixed")] = bc["bootstrap_fixed"]
for ds in ["colon", "leukemia"]:
    D[(ds, "bootstrap_fixed")] = fin[ds]["bootstrap_fixed"]
for ds in ["breast_cancer", "colon", "leukemia"]:
    D[(ds, "seed_control")] = mat[ds]          # matched control

A = {}
for (ds, c), blob in D.items():
    Zf, Zp, accs, ks = masks_from_raw(blob["raw_runs"], P[ds])
    jf = pairwise_jaccard(Zf); jp = pairwise_jaccard(Zp)
    sf, _ = nogueira_stability(Zf); sp, _ = nogueira_stability(Zp)
    # all pairwise jaccard values (for distribution plot)
    Zb = Zf.astype(bool); vals = []
    for i in range(len(Zb)):
        for j in range(i + 1, len(Zb)):
            un = np.logical_or(Zb[i], Zb[j]).sum()
            vals.append(np.logical_and(Zb[i], Zb[j]).sum() / un if un else 0.0)
    A[(ds, c)] = dict(nog_f=sf, nog_p=sp, jf=jf, jp=jp, accs=accs, ks=ks,
                      jac_all=np.array(vals), full_fronts=blob["raw_runs"])

_masks_cache = {}
for (ds_, c_), blob_ in D.items():
    Zf_, _, _, _ = masks_from_raw(blob_["raw_runs"], P[ds_])
    _masks_cache[(ds_, c_)] = Zf_

DS = ["breast_cancer", "colon", "leukemia"]
x = np.arange(len(DS)); w = 0.36

# ================= FIG 1: Nogueira stability across datasets =================
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, lvl, title in zip(axes, ["nog_f", "nog_p"],
                           ["Front-level", "Point-level (knee)"]):
    b = [A[(d, "bootstrap_fixed")][lvl] for d in DS]
    s = [A[(d, "seed_control")][lvl] for d in DS]
    ax.bar(x - w/2, b, w, label=COND["bootstrap_fixed"], color=C_BOOT)
    ax.bar(x + w/2, s, w, label=COND["seed_control"], color=C_SEED)
    ax.set_xticks(x); ax.set_xticklabels([NICE[d] for d in DS], fontsize=8)
    ax.set_ylabel("Nogueira stability"); ax.set_title(title, fontsize=9)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_ylim(-0.03, 0.58)
    for xi, v in zip(x - w/2, b):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.5)
    for xi, v in zip(x + w/2, s):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.5)
axes[0].legend(fontsize=7.5, frameon=False, loc="lower center",
               bbox_to_anchor=(1.1, -0.42), ncol=2)
fig.savefig("fig1_stability.png"); fig.savefig("fig1_stability.pdf"); plt.close(fig)

# ============ FIG 2: pairwise Jaccard distributions (the key figure) ============
fig, ax = plt.subplots(figsize=(7.2, 3.1))
pos, labels, colors = [], [], []
for i, d in enumerate(DS):
    for j, c in enumerate(["bootstrap_fixed", "seed_control"]):
        pos.append(i * 2.6 + j * 0.85)
        short = {"breast_cancer":"Breast\nCancer","colon":"Colon","leukemia":"Leukemia"}[d]
        labels.append(short + ("\n[data]" if j == 0 else "\n[search]"))
        colors.append(C_BOOT if j == 0 else C_SEED)
data = [A[(d, c)]["jac_all"] for d in DS for c in ["bootstrap_fixed", "seed_control"]]
bp = ax.boxplot(data, positions=pos, widths=0.62, patch_artist=True,
                medianprops=dict(color="black", lw=1.4), showfliers=False)
# chance baseline: mean Jaccard for random subsets of the SAME sizes.
# Without this the figure overstates the effect, since expected overlap
# depends strongly on p (30 vs 2000 vs 7070).
_rngc = np.random.default_rng(1)
for p_, (d_, c_) in zip(pos, [(d, c) for d in DS for c in ["bootstrap_fixed", "seed_control"]]):
    Zc = np.array([m for m in _masks_cache[(d_, c_)]])
    ksc = Zc.sum(1); pp = P[d_]
    vals = []
    for _ in range(3000):
        i, j = _rngc.choice(len(Zc), 2, replace=False)
        a1 = _rngc.choice(pp, ksc[i], replace=False)
        a2 = _rngc.choice(pp, ksc[j], replace=False)
        inter = len(np.intersect1d(a1, a2))
        vals.append(inter / (ksc[i] + ksc[j] - inter))
    ax.plot([p_ - 0.31, p_ + 0.31], [np.mean(vals)] * 2, color="#7a3b8f",
            lw=1.8, ls="--", zorder=5,
            label="Chance level (random subsets, same sizes)" if p_ == pos[0] else None)
ax.legend(fontsize=7, frameon=False, loc="upper right")
for patch, col in zip(bp["boxes"], colors):
    patch.set_facecolor(col); patch.set_alpha(0.75); patch.set_edgecolor("black"); patch.set_linewidth(0.6)
rng = np.random.default_rng(0)
for p_, dd in zip(pos, data):
    ax.scatter(p_ + rng.normal(0, 0.09, len(dd)), dd, s=5, color="black", alpha=0.30, zorder=3)
ax.set_xticks(pos); ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Pairwise Jaccard similarity\nbetween runs (front-level)")
ax.axhline(0, color="k", lw=0.6)
fig.savefig("fig2_jaccard.png"); fig.savefig("fig2_jaccard.pdf"); plt.close(fig)

# ============ FIG 3: accuracy vs #features (all Pareto points) ============
fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7), sharey=True)
for ax, d in zip(axes, DS):
    for c, col in [("bootstrap_fixed", C_BOOT), ("seed_control", C_SEED)]:
        xs, ys = [], []
        for run in A[(d, c)]["full_fronts"]:
            for sub in _front_of(run):
                if sub["n_sel"] > 0:
                    xs.append(sub["n_sel"]); ys.append(sub["acc"])
        ax.scatter(xs, ys, s=13, alpha=0.55, color=col, edgecolors="none",
                   label=COND[c] if d == DS[0] else None)
    ax.set_title(NICE[d].replace("\n", " "), fontsize=8.5)
    ax.set_xlabel("Number of selected features")
axes[0].set_ylabel("Inner-CV balanced accuracy")
axes[0].legend(fontsize=7, frameon=False, loc="lower right")
fig.savefig("fig3_acc_vs_size.png"); fig.savefig("fig3_acc_vs_size.pdf"); plt.close(fig)

# ==== FIG 4: the headline -- accuracy high while agreement is zero ====
fig, ax1 = plt.subplots(figsize=(6.4, 3.0))
# Use OUT-OF-BAG accuracy, not the optimized inner-CV value. The optimized
# figure is the quantity NSGA-II maximized and is optimistically biased
# (by 17-22 points on the microarray datasets); showing it here would
# contradict Table V.
_oob = json.load(open("oob_results.json"))
acc_med = [_oob[d]["oob_med"] for d in DS]
jac_med = [A[(d, "bootstrap_fixed")]["jf"]["median"] for d in DS]
ax1.bar(x - w/2, acc_med, w, color="#4a7c59", label="Median held-out (OOB) accuracy")
ax1.set_ylabel("Median held-out balanced accuracy", color="#4a7c59")
ax1.tick_params(axis="y", labelcolor="#4a7c59"); ax1.set_ylim(0, 1.18)
ax2 = ax1.twinx()
ax2.bar(x + w/2, jac_med, w, color="#b5651d", label="Median pairwise Jaccard")
# chance baseline for Jaccard, per dataset (depends on p and subset size).
# Without this the comparison across datasets is confounded by feature-space size.
_rng4 = np.random.default_rng(2)
for xi, d in zip(x + w/2, DS):
    Zc = _masks_cache[(d, "bootstrap_fixed")]; ksc = Zc.sum(1); pp = P[d]
    vv = []
    for _ in range(3000):
        i, j = _rng4.choice(len(Zc), 2, replace=False)
        a1 = _rng4.choice(pp, ksc[i], replace=False); a2 = _rng4.choice(pp, ksc[j], replace=False)
        it = len(np.intersect1d(a1, a2)); vv.append(it / (ksc[i] + ksc[j] - it))
    ax2.plot([xi - w/2, xi + w/2], [np.mean(vv)] * 2, color="#7a3b8f", lw=1.8, ls="--",
             zorder=6, label="Chance level (Jaccard)" if d == DS[0] else None)
ax2.set_ylabel("Median pairwise Jaccard", color="#b5651d")
ax2.tick_params(axis="y", labelcolor="#b5651d"); ax2.set_ylim(0, 1.18)
ax2.grid(False)
ax1.set_xticks(x); ax1.set_xticklabels([NICE[d] for d in DS], fontsize=8)
for xi, v in zip(x - w/2, acc_med):
    ax1.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)
for xi, v in zip(x + w/2, jac_med):
    ax2.text(xi, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)
h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1 + h2, l1 + l2, fontsize=7.5, frameon=False,
           loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=2)
fig.savefig("fig4_headline.png"); fig.savefig("fig4_headline.pdf"); plt.close(fig)

print("Figures written: fig1..fig4 (.png and .pdf)")
for d in DS:
    a = A[(d, "bootstrap_fixed")]
    print(f"  {d:>14}: nog_front={a['nog_f']:.4f}  jac_med={a['jf']['median']:.4f}  "
          f"acc_med={np.median(a['accs']):.4f}  k_med={np.median(a['ks']):.1f}")
