import time, json, os, sys
sys.path.insert(0,'.')
import bootstrap as bs, numpy as np

OUT="matched_seed_control.json"
out=json.load(open(OUT)) if os.path.exists(OUT) else {}

def clean(r, seeds):
    d={k:v for k,v in r.items() if k!="fronts"}
    d["front_sizes"]=[len(f) for f in r["fronts"]]
    d["seeds"]=seeds
    d["raw_runs"]=[{"seed":seeds[i],
        "front":[{"n_sel":int(k),"acc":float(a),"features":np.where(m)[0].tolist()}
                 for (m,a,k) in f]} for i,f in enumerate(r["fronts"])]
    return d

t0=time.time()
for ds in ["breast_cancer","colon","leukemia"]:
    if ds in out:
        print(f"SKIP {ds} (done)",flush=True); continue
    print(f"===== {ds} MATCHED seed control =====",flush=True)
    r=bs.run_seed_control_matched(ds, n_seeds=10, boot_seed=0, base_seed=100,
                                   pop_size=40, n_gen=20)
    out[ds]=clean(r, list(range(100,110)))
    json.dump(out, open(OUT,"w"), indent=2, default=str)
    print(f"--- CHECKPOINT {ds}: front={r['front_level']['stability']:.4f} "
          f"point={r['point_level']['stability']:.4f} cv={r['cv_scheme']}({r['n_folds']}) "
          f"[{time.time()-t0:.0f}s] ---",flush=True)
print("ALL DONE",time.time()-t0,flush=True)
