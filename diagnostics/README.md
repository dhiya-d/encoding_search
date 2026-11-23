# diagnostics/ — amplitude-encoding diagnostics (paper Sec. V A, Fig. 2)

Each script loops over datasets and the three preprocessing modes (`uncentered`,
`centered`, `whitened`; see `core/diag_common.py`).

| Script | Question | Output |
|---|---|---|
| `diag_a_phase_check.py` | Is a chance-level linear probe on encoded states a global-sign artefact? Compares six readouts on the same images. Also used to encode the test split with any circuit (`--genome-root`, `--genome-json`; default: the hand-designed reference). | `<out>/<ds>/a_phase_check/<mode>/` incl. `encoded_states.npz` |
| `diag_b_ceiling.py` | What can the exact amplitude states reach (perfect circuit), compared with classical models on the same and on richer inputs? Kernel-target alignment with a permutation null. | `experiments/<ds>/b_ceiling/` |
| `diag_c_spectral.py` | Where does the class signal sit (pixel, Fourier, Walsh bases), and how much of it can survive a given infidelity (class-signal budget)? | `<out>/<ds>/c_spectral/<mode>/` |

```bash
python diagnostics/diag_b_ceiling.py
python diagnostics/diag_a_phase_check.py --out experiments_handref
python diagnostics/diag_c_spectral.py    --out experiments_handref
```

Common options: `--datasets`, `--modes`, `--data-root` (default `data`), `--out`, `--force`.
Each script prints a plain-language verdict and stores it in `results.json`.
