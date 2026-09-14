import time, json, sys
sys.path.insert(0, '.')
import bootstrap as bs

t0 = time.time()
res = bs.run_bootstrap_study("leukemia", n_bootstraps=10, seed=0, pop_size=40, n_gen=20, verbose=True)
res_clean = {k: v for k, v in res.items() if k != "fronts"}
res_clean["front_sizes"] = [len(f) for f in res["fronts"]]
with open("main_run_result.json", "w") as f:
    json.dump(res_clean, f, indent=2, default=str)
print("TOTAL TIME", time.time() - t0)
print("SAVED main_run_result.json")
