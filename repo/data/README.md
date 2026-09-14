# Data

`colon.mat` and `leukemia.mat` are **not included** in this repository
because they are third-party datasets with their own distribution terms
(Alon et al. 1999; Golub et al. 1999, via the scikit-feature repository).

## Where to get them

Download from the scikit-feature data repository:
https://jundongl.github.io/scikit-feature/datasets.html

Place them here as:
```
data/colon.mat
data/leukemia.mat
```

`src/data.py` expects exactly these two filenames in this folder.

Breast Cancer Wisconsin is loaded directly from `sklearn.datasets`, so it
needs no manual download.
