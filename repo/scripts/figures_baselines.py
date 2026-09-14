"""
Figures 5 and 6 — the baseline comparisons added in Phase 1 and Phase 2.
Reads only saved result files; runs no experiment.
"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, '.')
from stability_ci import nogueira_stability

P = {"breast_cancer": 30, "colon": 2000, "leukemia": 7070}
NICE = {"breast_cancer": "Breast Cancer\n(p/n=0.05)", "colon": "Colon\n(p/n=32)",
        "leukemia": "Leukemia\n(p/n=98)"}
DS = ["breast_cancer", "colon", "leukemia"]
C_NSGA, C_RS, C_MRMR = "#3b6ea5", "#8a8a8a", "#b5651d"

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 300, "savefig.bbox": "tight", "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})

rs   = json.load(open("phase1_random_search.json"))
ver  = json.load(open("phase1_nsga_verification.json"))
mr   = json.load(open("phase2_mrmr.json"))
bc   = json.load(open("breast_cancer_matched_result.json"))
fin  = json.load(open("final_raw_results.json"))
mat  = json.load(open("matched_seed_control.json"))
bj   = json.load(open("baseline_jaccard.json"))
ma   = json.load(open("matched_accuracy.json"))


def _front(r):
    return r["front"] if isinstance(r, dict) and "front" in r else r


def phi_front(runs, p):
    Z = []
    for r in runs:
        f = [s for s in _front(r) if s["n_sel"] > 0]
        if not f:
            continue
        u = np.zeros(p, bool)
        for s in f:
            u[s["features"]] = True
        Z.append(u)
    v, _ = nogueira_stability(np.array(Z))
    return v


# ============ FIG 5: NSGA-II vs budget-matched random search ============
fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
for ax, cond, title in zip(axes, ["bootstrap", "fixed"],
                            ["Data resampling", "Search randomness only"]):
    n_vals, r_vals = [], []
    for d in DS:
        nsrc = ((bc if d == "breast_cancer" else fin[d])["bootstrap_fixed"]["raw_runs"]
                if cond == "bootstrap" else mat[d]["raw_runs"])
        n_vals.append(phi_front(nsrc, P[d]))
        r_vals.append(phi_front(rs[f"{d}|{cond}"]["runs"], P[d]))
    x = np.arange(3); w = 0.36
    ax.bar(x - w/2, n_vals, w, color=C_NSGA, label="NSGA-II")
    ax.bar(x + w/2, r_vals, w, color=C_RS, label="Random search\n(same budget)")
    ax.set_xticks(x); ax.set_xticklabels([NICE[d] for d in DS], fontsize=7.5)
    ax.set_title(title, fontsize=9); ax.set_ylabel("Nogueira stability")
    ax.axhline(0, color="k", lw=0.6); ax.set_ylim(-0.02, 0.58)
    for xi, v in zip(x - w/2, n_vals):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.5)
    for xi, v in zip(x + w/2, r_vals):
        ax.text(xi, v + 0.012, f"{v:.3f}", ha="center", fontsize=6.5)
axes[0].legend(fontsize=7, frameon=False, loc="upper center",
               bbox_to_anchor=(1.1, -0.30), ncol=2)
fig.savefig("fig5_random_search.png"); fig.savefig("fig5_random_search.pdf"); plt.close(fig)

# ==== FIG 6: cardinality-matched NSGA-II vs mRMR, with distinct-subset counts ====
fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9), sharey=True)
ks = [2, 4, 8]
for ax, d in zip(axes, DS):
    n_v = [json.load(open("cardinality_matched.json"))[d][str(k)]["nsga_phi"] for k in ks]
    m_v = [json.load(open("cardinality_matched.json"))[d][str(k)]["mrmr_phi"] for k in ks]
    x = np.arange(3); w = 0.36
    ax.bar(x - w/2, n_v, w, color=C_NSGA, label="NSGA-II")
    ax.bar(x + w/2, m_v, w, color=C_MRMR, label="mRMR")
    # annotate how many DISTINCT subsets each method produced in 10 runs
    for xi, k in zip(x, ks):
        ax.text(xi, -0.078, f"{bj[d][str(k)]['nsga_distinct']}/10", ha="center",
                fontsize=6, color=C_NSGA)
        ax.text(xi, -0.108, f"{bj[d][str(k)]['mrmr_distinct']}/10", ha="center",
                fontsize=6, color=C_MRMR)
    ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in ks], fontsize=8)
    ax.set_title(NICE[d].replace("\n", " "), fontsize=8.5)
    ax.axhline(0, color="k", lw=0.6); ax.set_ylim(-0.15, 1.08)
axes[0].set_ylabel("Nogueira stability")
axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
fig.text(0.5, -0.10, "below each pair: distinct feature subsets in 10 bootstraps, NSGA-II (blue) over mRMR (orange).  "
                     "10/10 = the method never repeated a selection", ha="center", fontsize=7)
fig.savefig("fig6_mrmr.png"); fig.savefig("fig6_mrmr.pdf"); plt.close(fig)

print("wrote fig5, fig6 (png + pdf)")
for d in DS:
    print(f"  {d:>14}: NSGA phi(k=2)={json.load(open('cardinality_matched.json'))[d]['2']['nsga_phi']:.4f} "
          f"mRMR={json.load(open('cardinality_matched.json'))[d]['2']['mrmr_phi']:.4f}")
