import time, json, sys
sys.path.insert(0, '.')
import bootstrap as bs

t0 = time.time()
print("=== COLON: bootstrap study (10 bootstraps, same protocol as Leukemia) ===", flush=True)
res_boot = bs.run_bootstrap_study("colon", n_bootstraps=10, seed=0, pop_size=40, n_gen=20)
print(f"[bootstrap done at {time.time()-t0:.0f}s]", flush=True)

print("=== COLON: seed-only control (10 seeds) ===", flush=True)
res_seed = bs.run_seed_control_study("colon", n_seeds=10, base_seed=100, pop_size=40, n_gen=20)
print(f"[seed control done at {time.time()-t0:.0f}s]", flush=True)

def clean(r):
    d = {k: v for k, v in r.items() if k != "fronts"}
    d["front_sizes"] = [len(f) for f in r["fronts"]]
    return d

out = {"bootstrap_fixed": clean(res_boot), "seed_control": clean(res_seed)}
with open("colon_result.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("TOTAL TIME", time.time() - t0, flush=True)
print("SAVED colon_result.json", flush=True)
