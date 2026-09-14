import time, json, sys
sys.path.insert(0, '.')
import bootstrap as bs

t0 = time.time()
print("=== PART 1: seed-only control (same data, varying NSGA-II seed) ===", flush=True)
res_seed = bs.run_seed_control_study("leukemia", n_seeds=10, base_seed=100, pop_size=40, n_gen=20)
print(f"[part1 done at {time.time()-t0:.0f}s]", flush=True)

print("=== PART 2: bootstrap variance, WITH duplicate-leakage fix ===", flush=True)
res_boot = bs.run_bootstrap_study("leukemia", n_bootstraps=10, seed=0, pop_size=40, n_gen=20)
print(f"[part2 done at {time.time()-t0:.0f}s]", flush=True)

def clean(r):
    d = {k: v for k, v in r.items() if k != "fronts"}
    d["front_sizes"] = [len(f) for f in r["fronts"]]
    return d

out = {"seed_control": clean(res_seed), "bootstrap_fixed": clean(res_boot)}
with open("control_vs_bootstrap_result.json", "w") as f:
    json.dump(out, f, indent=2, default=str)
print("TOTAL TIME", time.time() - t0, flush=True)
print("SAVED control_vs_bootstrap_result.json", flush=True)
