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

fin=json.load(open('final_raw_results.json'))
bc=json.load(open('breast_cancer_matched_result.json'))
def fr(r): return r['front'] if isinstance(r,dict) else r

out={}
for ds in ['breast_cancer','colon','leukemia']:
    X,y,tag=dm.load(ds)
    rng=np.random.default_rng(0)
    runs = bc['bootstrap_fixed']['raw_runs'] if ds=='breast_cancer' else fin[ds]['bootstrap_fixed']['raw_runs']
    rep,oob,ks=[],[],[]
    for b in range(10):
        bidx=bs.stratified_bootstrap_indices(y,rng)
        front=fr(runs[b])
        front=[s for s in front if s['n_sel']>0]
        if not front: continue
        kp=knee(front); feats=kp['features']
        oidx=np.setdiff1d(np.arange(len(y)),np.unique(bidx))
        if len(oidx)<5 or len(np.unique(y[oidx]))<2: continue
        Xtr,ytr=X[np.ix_(bidx,feats)],y[bidx]
        Xte,yte=X[np.ix_(oidx,feats)],y[oidx]
        sc=StandardScaler(); Xtr=sc.fit_transform(Xtr); Xte=sc.transform(Xte)
        clf=KNeighborsClassifier(n_neighbors=min(5,len(ytr))); clf.fit(Xtr,ytr)
        rep.append(kp['acc']); oob.append(bal(yte,clf.predict(Xte))); ks.append(kp['n_sel'])
    out[ds]=dict(rep_med=float(np.median(rep)), oob_med=float(np.median(oob)),
                 oob_min=float(np.min(oob)), oob_max=float(np.max(oob)),
                 oob_q1=float(np.percentile(oob,25)), oob_q3=float(np.percentile(oob,75)),
                 k_med=float(np.median(ks)), n=len(oob))
    o=out[ds]
    print(f"{ds:>14}: reported {o['rep_med']:.3f} -> OOB {o['oob_med']:.3f} "
          f"[IQR {o['oob_q1']:.3f}-{o['oob_q3']:.3f}, range {o['oob_min']:.3f}-{o['oob_max']:.3f}] n={o['n']}")
json.dump(out,open('oob_results.json','w'),indent=2)
