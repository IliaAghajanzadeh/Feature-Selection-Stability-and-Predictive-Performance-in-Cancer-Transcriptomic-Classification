import json, numpy as np, sys
sys.path.insert(0,'.')
from stability_ci import pairwise_jaccard, nogueira_stability

P = {"breast_cancer":30, "colon":2000, "leukemia":7070}

def masks_from_raw(raw_runs, p):
    """Rebuild front-level (union) and point-level (knee) masks per run."""
    front_masks, point_masks, accs, ks = [], [], [], []
    for run in raw_runs:
        front = run["front"] if isinstance(run, dict) and "front" in run else run
        u = np.zeros(p, dtype=int)
        for sub in front:
            u[sub["features"]] = 1
        front_masks.append(u)
        if not front: continue
        a = np.array([s["acc"] for s in front]); k = np.array([s["n_sel"] for s in front])
        ar, kr = a.max()-a.min(), k.max()-k.min()
        an = (a-a.min())/ar if ar>0 else np.zeros_like(a)
        kn = (k-k.min())/kr if kr>0 else np.zeros_like(k)
        knee = front[int(np.argmin(np.sqrt((1-an)**2 + kn**2)))]
        pm = np.zeros(p, dtype=int); pm[knee["features"]] = 1
        point_masks.append(pm)
        accs.append(knee["acc"]); ks.append(knee["n_sel"])
    return np.array(front_masks), np.array(point_masks), accs, ks

rows = []
bc = json.load(open("breast_cancer_matched_result.json"))
fin = json.load(open("final_raw_results.json"))
srcs = {("breast_cancer",c): bc[c] for c in ["bootstrap_fixed","seed_control"]}
for ds in ["colon","leukemia"]:
    for c in ["bootstrap_fixed","seed_control"]:
        srcs[(ds,c)] = fin[ds][c]

for (ds, cond), blob in srcs.items():
    p = P[ds]
    raw = blob.get("raw_runs")
    Zf, Zp, accs, ks = masks_from_raw(raw, p)
    sf,_ = nogueira_stability(Zf); sp,_ = nogueira_stability(Zp)
    jf = pairwise_jaccard(Zf); jp = pairwise_jaccard(Zp)
    rows.append(dict(dataset=ds, condition=cond,
        nog_front=round(sf,4), nog_point=round(sp,4),
        jac_front_med=round(jf["median"],4), jac_front_iqr=f'[{jf["q1"]:.3f},{jf["q3"]:.3f}]',
        jac_point_med=round(jp["median"],4), jac_point_iqr=f'[{jp["q1"]:.3f},{jp["q3"]:.3f}]',
        jac_point_max=round(jp["max"],3),
        acc_med=round(float(np.median(accs)),4), k_med=float(np.median(ks)),
        k_min=int(np.min(ks)), k_max=int(np.max(ks))))

json.dump(rows, open("final_analysis.json","w"), indent=2)
hdr = ["dataset","condition","nog_front","nog_point","jac_front_med","jac_point_med","jac_point_max","acc_med","k_med","k_min","k_max"]
print(" | ".join(f"{h:>14}" for h in hdr))
for r in rows:
    print(" | ".join(f"{str(r[h]):>14}" for h in hdr))
print()
for r in rows:
    print(f'{r["dataset"]}/{r["condition"]}: jaccard front IQR {r["jac_front_iqr"]}  point IQR {r["jac_point_iqr"]}')
