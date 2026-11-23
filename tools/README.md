# tools/

* `prepare_dataset.py` downloads a MedMNIST dataset into `data/<name>/` in the format the
  pipeline expects, with optional stratified caps on the training and validation splits:
  `python tools/prepare_dataset.py pathmnist --max-train 20000 --max-val 5000`.
* `pennylane_encoding.py` rebuilds an exported Fourier encoding (and its trained
  re-uploading classifier) as a PennyLane circuit, e.g. to run it on hardware or another
  simulator, and checks it against the NumPy simulator:
  `python tools/pennylane_encoding.py experiments_fourier/bloodmnist/downstream/encodings/search_best.json`.
