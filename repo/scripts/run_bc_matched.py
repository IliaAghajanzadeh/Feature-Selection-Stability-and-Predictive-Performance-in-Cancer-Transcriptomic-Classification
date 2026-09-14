import time, json, sys
sys.path.insert(0, '.')
import bootstrap as bs, numpy as np

def clean(r):
    d = {k: v for k, v in r.items() if k != "fronts"}
    d["front_sizes"] = [len(f) for f in r["fronts"]]
    # SAVE RAW per-run detail this time (audit finding #2)
    d["raw_runs"] = [
        [{"n_sel": int(k), "acc": float(a), "features": np.where(msk)[0].tolist()}
         for (msk, a, k) in front]
        for front in r["fronts"]
    ]
    return d

t0 = time.time()
print("=== BREAST CANCER, MATCHED PROTOCOL (pop=40, gen=20, 10 runs, post-fix) ===", flush=True)
res_boot = bs.run_bootstrap_study("breast_cancer", n_bootstraps=10, seed=0, pop_size=40, n_gen=20)
print(f"[bootstrap done {time.time()-t0:.0f}s]", flush=True)
res_seed = bs.run_seed_control_study("breast_cancer", n_seeds=10, base_seed=100, pop_size=40, n_gen=20)
print(f"[seed control done {time.time()-t0:.0f}s]", flush=True)

out = {"bootstrap_fixed": clean(res_boot), "seed_control": clean(res_seed),
       "protocol": {"pop_size":40,"n_gen":20,"n_runs":10,"post_leakage_fix":True}}
json.dump(out, open("breast_cancer_matched_result.json","w"), indent=2, default=str)
print("front-level boot:", res_boot["front_level"]["stability"])
print("point-level boot:", res_boot["point_level"]["stability"])
print("front-level seed:", res_seed["front_level"]["stability"])
print("point-level seed:", res_seed["point_level"]["stability"])
print("SAVED breast_cancer_matched_result.json", flush=True)
