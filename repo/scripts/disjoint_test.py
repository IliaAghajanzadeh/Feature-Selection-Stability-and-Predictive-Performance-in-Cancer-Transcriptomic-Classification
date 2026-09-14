"""
Do DISJOINT subsets from different runs achieve comparable held-out accuracy?

Uses the MATCHED seed-control condition: all 10 runs share ONE fixed bootstrap
draw, hence one common out-of-bag set. Every subset is therefore evaluated on
IDENTICAL held-out data -- the only clean way to compare subsets against each
other.
"""
import json, numpy as np, sys
sys.path.insert(0,'.')
import data as dm, bootstrap as bs
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier

def bal(yt,yp):
    cs=np.unique(yt)
    return float(np.mean([(yp[yt==c]==c).mean() for c in cs])) if len(cs)>1 else float((yp==yt).mean())

def knee(front):
    a=np.array([s['acc'] for s in front]); k=np.array([s['n_sel'] for s in front])
    ar,kr=a.max()-a.min(),k.max()-k.min()
    an=(a-a.min())/ar if ar>0 else np.zeros_like(a)
    kn=(k-k.min())/kr if kr>0 else np.zeros_like(k)
    return front[int(np.argmin(np.sqrt((1-an)**2+kn**2)))]

mat=json.load(open('matched_seed_control.json'))
out={}
for ds in ['breast_cancer','colon','leukemia']:
    X,y,tag=dm.load(ds)
    rng=np.random.default_rng(0)
    bidx=bs.stratified_bootstrap_indices(y,rng)          # the ONE fixed draw
    oob=np.setdiff1d(np.arange(len(y)),np.unique(bidx))   # common held-out set

    subsets, accs = [], []
    for run in mat[ds]['raw_runs']:
        front=[s for s in run['front'] if s['n_sel']>0]
        if not front: continue
        f=knee(front)['features']
        Xtr,ytr=X[np.ix_(bidx,f)],y[bidx]
        Xte,yte=X[np.ix_(oob,f)],y[oob]
        sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
        clf=KNeighborsClassifier(n_neighbors=min(5,len(ytr))); clf.fit(Xtr,ytr)
        subsets.append(set(f)); accs.append(bal(yte,clf.predict(Xte)))
    accs=np.array(accs)

    dis_d, ov_d = [], []
    for i in range(len(subsets)):
        for j in range(i+1,len(subsets)):
            d=abs(accs[i]-accs[j])
            (dis_d if subsets[i].isdisjoint(subsets[j]) else ov_d).append(d)
    out[ds]=dict(n_runs=len(subsets), oob_n=int(len(oob)),
                 acc_med=float(np.median(accs)), acc_min=float(accs.min()), acc_max=float(accs.max()),
                 acc_iqr=[float(np.percentile(accs,25)),float(np.percentile(accs,75))],
                 n_disjoint=len(dis_d), n_overlap=len(ov_d),
                 disjoint_med_diff=float(np.median(dis_d)) if dis_d else None,
                 disjoint_max_diff=float(np.max(dis_d)) if dis_d else None,
                 overlap_med_diff=float(np.median(ov_d)) if ov_d else None)
    o=out[ds]
    print(f"=== {tag} (common OOB set, n={o['oob_n']}) ===")
    print(f"  {o['n_runs']} subsets, held-out acc: median {o['acc_med']:.3f}, "
          f"IQR {o['acc_iqr'][0]:.3f}-{o['acc_iqr'][1]:.3f}, range {o['acc_min']:.3f}-{o['acc_max']:.3f}")
    print(f"  disjoint pairs: {o['n_disjoint']}/{o['n_disjoint']+o['n_overlap']}")
    if dis_d:
        print(f"  |acc difference| among DISJOINT pairs: median {o['disjoint_med_diff']:.3f}, max {o['disjoint_max_diff']:.3f}")
    if ov_d:
        print(f"  |acc difference| among OVERLAPPING pairs: median {o['overlap_med_diff']:.3f}")
    print()
json.dump(out,open('disjoint_test.json','w'),indent=2)
