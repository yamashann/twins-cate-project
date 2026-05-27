# Data directory

Raw data are downloaded automatically by `src/data.py` from
[AMLab-Amsterdam/CEVAE](https://github.com/AMLab-Amsterdam/CEVAE/tree/master/datasets/TWINS)
the first time any script needs them, and cached under `data/raw/`.

If you prefer to download manually, place the following files here:

```
data/raw/twin_pairs_X_3years_samesex.csv
data/raw/twin_pairs_T_3years_samesex.csv
data/raw/twin_pairs_Y_3years_samesex.csv
data/raw/covar_desc.txt
data/raw/covar_type.txt
```

`data/raw/` is gitignored.
