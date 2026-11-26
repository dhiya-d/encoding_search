# search_frobenius/ — Search 1: the reconstruction search

The first of the three amplitude-encoder searches, and the one whose result
motivates the other two (thesis §3.8.2).

Every 16×16 grayscale image is prepared by its own fitted circuit, exactly
as in `search_amplitude/`. NSGA-II searches the circuit **structure** over
two objectives:

1. **mean Frobenius residual** $\lVert t - \psi \rVert_2$ between the target
   and fitted amplitude vectors (minimised);
2. **number of two-qubit gates** (minimised).

A single winner is then taken off the front by minimising
`residual + 0.004 × gates` within a knee tolerance of `0.015`.

Settings used for the reported runs, recorded in each `search.log`:

| | |
|---|---|
| Population / generations | 24 / 12 |
| Per-image fitting | 150 Adam steps, lr 0.08 |
| Batch (search / re-benchmark) | 256 / 512 |
| Seeds | 0–4 |
| Datasets | blood-, breast-, derma-, retinamnist (16×16 grayscale) |

Unlike `search_amplitude/`, this search space admits controlled-$R_Y$ gates
and is not capped at 64 two-qubit gates; selected circuits run to 73.

## Outputs

`results/frobenius/<dataset>/`

| Path | Content |
|---|---|
| `seed<s>/search_result.json` | the full front, the winner, and all settings |
| `seed<s>/best_genome.json` | the selected architecture |
| `seed<s>/search.log` | per-generation trace: best residual, front size, gate range |
| `seed<s>/sweep.log` | every front member re-fitted and scored by the linear probe |
| `seed<s>/front.png`, `cost_signal_curve.*` | plots |
| `aggregate/aggregate_bins.csv` | residual and probe accuracy binned by gate count — the saturation result of thesis §3.8.2 |
| `qcnn/` | QCNN trained on the selected circuit's encoded states |

## Status of the code

> **Note.** The search script for this study is not yet in the repository.
> The runs were produced with an earlier codebase; every setting needed to
> reproduce them is recorded at the top of each `search.log` and in the
> `search` block of each `search_result.json`. Replace this note once the
> script is restored.
