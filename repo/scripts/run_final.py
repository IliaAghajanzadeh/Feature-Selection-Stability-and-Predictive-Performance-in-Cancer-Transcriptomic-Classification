import time, json, os, sys
sys.path.insert(0, '.')
import bootstrap as bs, numpy as np

OUT = "final_raw_results.json"

def clean(r, seeds):
    d = {k: v for k, v in r.items() if k != "fronts"}
    d["front_sizes"] = [len(f) for f in r["fronts"]]
    d["seeds"] = seeds
    d["raw_runs"] = [
        {"seed": seeds[i],
         "front": [{"n_sel": int(k), "acc": float(a), "features": np.where(msk)[0].tolist()}
                   for (msk, a, k) in front]}
        for i, front in enumerate(r["fronts"])]
    return d

out = json.load(open(OUT)) if os.path.exists(OUT) else {}

jobs = [
    ("colon",    "bootstrap_fixed"), ("colon",    "seed_control"),
    ("leukemia", "bootstrap_fixed"), ("leukemia", "seed_control"),
]
t0 = time.time()
for name, cond in jobs:
    if out.get(name, {}).get(cond):
        print(f"SKIP {name}/{cond} (already done)", flush=True); continue
    print(f"===== {name} / {cond} =====", flush=True)
    if cond == "bootstrap_fixed":
        r = bs.run_bootstrap_study(name, n_bootstraps=10, seed=0, pop_size=40, n_gen=20)
        seeds = list(range(0,10))
    else:
        r = bs.run_seed_control_study(name, n_seeds=10, base_seed=100, pop_size=40, n_gen=20)
        seeds = list(range(100,110))
    out.setdefault(name, {})[cond] = clean(r, seeds)
    json.dump(out, open(OUT,"w"), indent=2, default=str)
    print(f"--- CHECKPOINT saved after {name}/{cond} at {time.time()-t0:.0f}s ---", flush=True)

out["protocol"] = {"pop_size":40,"n_gen":20,"n_runs":10,"classifier":"kNN k=5","post_leakage_fix":True}
json.dump(out, open(OUT,"w"), indent=2, default=str)
print("ALL DONE", time.time()-t0, flush=True)
