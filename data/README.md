# data/

The datasets are not part of the repository. Each dataset lives in its own folder:

```
data/<dataset>/{train,val,test}_images.npy    uint8, (N, 28, 28) or (N, 28, 28, 3)
data/<dataset>/{train,val,test}_labels.npy    int, (N,) or (N, 1)
```

Download them from MedMNIST v2 (Yang et al., Sci. Data 10, 41 (2023)) with

```bash
python run_all.py --stages data                      # every dataset used in the paper
python tools/prepare_dataset.py bloodmnist           # or one at a time
python tools/prepare_dataset.py pathmnist --max-train 20000 --max-val 5000
```

PathMNIST is capped to 20 000 training and 5 000 validation images (stratified,
seed 0); the test split is always complete. The other datasets are used in full.
