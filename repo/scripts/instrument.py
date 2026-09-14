"""
Measures TWO things the roadmap/prompt need, from the real code path:
  (1) actual number of fitness evaluations per run  [prompt 4-1: do not assume 840]
  (2) run-level feature exposure = unique features ever appearing in any
      EVALUATED individual                          [roadmap 0-8]
Instruments the real eval_fn; does not reimplement anything.
"""
import json, numpy as np, sys, time
sys.path.insert(0,'.')
import data as dm, nested_cv as ncv, ga as ga_mod, bootstrap as bs

out={}
for ds in ['breast_cancer','colon','leukemia']:
    X,y,tag=dm.load(ds); p=X.shape[1]
    rng=np.random.default_rng(0)
    bidx=bs.stratified_bootstrap_indices(y,rng)
    Xb,yb=X[bidx],y[bidx]
    cv=ncv.make_cv(len(yb),seed=0,groups=bidx)

    stats={'n_evals':0,'seen':np.zeros(p,bool),'k_sum':0}
    def eval_fn(mask, Xb=Xb, yb=yb, cv=cv, g=bidx, st=stats):
        st['n_evals']+=1
        st['seen'] |= mask
        st['k_sum']+=int(mask.sum())
        return ncv.eval_mask(Xb,yb,mask,cv,groups=g)

    t0=time.time()
    front=ga_mod.run_nsga2(p, eval_fn, pop_size=40, n_gen=20, seed=0)
    exposed=int(stats['seen'].sum())
    out[ds]=dict(p=p, n_evals=stats['n_evals'],
                 naive_840_assumption=40+40*20,
                 features_exposed=exposed,
                 exposure_pct=round(100*exposed/p,1),
                 mean_k_per_eval=round(stats['k_sum']/stats['n_evals'],1),
                 front_size=len(front), secs=round(time.time()-t0))
    print(f"{tag:>24}: evals={stats['n_evals']:>4} (naive assumption 840)  "
          f"exposed {exposed}/{p} = {out[ds]['exposure_pct']}%  "
          f"mean k/eval {out[ds]['mean_k_per_eval']}  [{out[ds]['secs']}s]", flush=True)
json.dump(out,open('instrumentation.json','w'),indent=2)
