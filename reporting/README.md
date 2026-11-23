# reporting/ — reports, tables and figures

| Script | Input | Output |
|---|---|---|
| `make_report.py` | all amplitude and Fourier results | `reports/report.md`, `reports/claims.json`, `reports/fig_*.png`: every claim of the paper with paired bootstrap 95% intervals (test images resampled, classifier seeds averaged) |
| `paper_tables.py` | `reports/*.json` | `reports/paper/tables.tex` (booktabs) and copies of the report figures |
| `paper_figures.py` | `reports/*.json` | `fig1_pipeline.pdf` … `fig6_noise.pdf` at REVTeX column width |

```bash
python reporting/make_report.py   --runs bloodmnist:centered pathmnist:centered breastmnist:uncentered
python reporting/paper_tables.py  --pairs bloodmnist:centered:bloodmnist pathmnist:centered:pathmnist
python reporting/paper_figures.py --dataset bloodmnist --out paper_figures/bloodmnist
```

`fig2_diagnostics.pdf` and `fig5_fourier.pdf` use values copied from the bloodmnist
reports into the dictionaries at the top of `paper_figures.py`; update them after a rerun.
