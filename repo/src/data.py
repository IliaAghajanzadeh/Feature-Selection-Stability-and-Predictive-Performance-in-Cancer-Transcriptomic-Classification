"""
Dataset loaders. Deliberately return RAW X, y with zero preprocessing.
All scaling/imputation must happen later, inside CV folds only (see nested_cv.py).
Loading raw here is itself part of leakage prevention: if preprocessing lived
here, it would be trivial to accidentally fit it on the full dataset once.
"""
import numpy as np
import scipy.io as sio
from pathlib import Path
from sklearn.datasets import load_breast_cancer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_breast_cancer_wisconsin():
    """Debug dataset. 569 samples, 30 features, binary, well-balanced-ish (357/212)."""
    d = load_breast_cancer()
    X = d.data.astype(float)
    y = d.target.astype(int)
    return X, y, "breast_cancer_wisconsin"


def _load_skfeature_mat(name):
    path = DATA_DIR / f"{name}.mat"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download from "
            f"https://raw.githubusercontent.com/jundongl/scikit-feature/master/skfeature/data/{name}.mat"
        )
    d = sio.loadmat(path)
    X = np.asarray(d["X"], dtype=float)
    y = np.asarray(d["Y"]).ravel()
    y = (y == y.max()).astype(int)  # map {-1,1} -> {0,1}; deterministic, not data-dependent stat
    return X, y


def load_leukemia():
    """ALLAML / Leukemia (Golub et al. 1999), via scikit-feature (ASU) repo.
    72 samples, 7070 genes, classes 47/25. p/n ~ 99 -> primary p>>n dataset."""
    X, y = _load_skfeature_mat("leukemia")
    return X, y, "leukemia_allaml"


def load_colon():
    """Colon (Alon et al. 1999), via scikit-feature (ASU) repo.
    62 samples, 2000 genes, classes 40/22. p/n ~ 32 -> optional robustness check."""
    X, y = _load_skfeature_mat("colon")
    return X, y, "colon"


REGISTRY = {
    "breast_cancer": load_breast_cancer_wisconsin,
    "leukemia": load_leukemia,
    "colon": load_colon,
}


def load(name):
    X, y, tag = REGISTRY[name]()
    assert X.shape[0] == y.shape[0], "X/y sample count mismatch"
    assert set(np.unique(y)) <= {0, 1}, "expected binary labels 0/1"
    return X, y, tag


if __name__ == "__main__":
    for k in REGISTRY:
        X, y, tag = load(k)
        n1 = int(y.sum())
        print(f"{tag:>22} | n={X.shape[0]:>4} p={X.shape[1]:>6} | class1={n1} class0={len(y)-n1}")
