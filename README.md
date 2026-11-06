# Fair Bubble Sort (FBS)

Code for the paper

> **Fair Bubble Sort: Provably Optimal Fair Ranking with Continuous Sensitive Attributes**  
> Antonio Ferrara, Fabio Vitale, Atsushi Miyauchi, Francesco Bonchi  
> Advances in Neural Information Processing Systems (NeurIPS 2026)

FBS is a greedy algorithm that re-ranks items to satisfy a fairness constraint on a continuous sensitive attribute while minimizing the weighted Kendall-tau distance to the original ranking. The core implementation uses Numba JIT compilation for high performance.

## Setup

```bash
pip install numpy numba matplotlib scipy pandas
```

For the baseline comparisons (DetConstSort and FA*IR):

```bash
pip install FairRankTune fairsearchcore
```

## Quick start

Run the toy example with randomly generated data:

```bash
python example.py
python example.py --tau 0.95 --n 20 --seed 7
```

Or run it on your own data via a CSV file with `score` and `sensitive` columns (a sample file is included):

```bash
python example.py --input sample_ranking.csv
python example.py --input sample_ranking.csv --tau 0.95
```

| Argument  | Description | Default |
|-----------|-------------|---------|
| `--tau`   | Fairness threshold in (0, 1]. Higher = stricter | 0.9 |
| `--input` | Path to a CSV with `score` and `sensitive` columns | (random) |
| `--n`     | Number of items (ignored if `--input` is provided) | 8 |
| `--seed`  | Random seed (ignored if `--input` is provided) | 4 |

## Repository structure

```
example.py                           # Toy example
sample_ranking.csv                   # Sample input for example.py
src/
  fbs_numba.py                       # Core FBS algorithm (Numba JIT)
  baselines.py                       # Baseline methods (DetConstSort, FA*IR, median split)
  experiment_approx_ratio.py         # Approximation ratio experiments
  experiment_sorted_case.py          # Sorted-case analysis
  experiment_five_block.py           # Five-block theorem verification
  experiment_milp_optimum.py         # MILP optimal solution comparison
  experiment_baselines_real.py       # Baselines comparison on real data
  experiment_baselines_synthetic.py  # Baselines comparison on synthetic data
  experiment_real_datasets.py        # Real dataset experiments
  experiment_scalability.py          # Scalability experiments
  experiment_scalability_numba.py    # Scalability with Numba acceleration
  experiment_scalability_baselines.py# Scalability of baseline methods
  plot_baselines_split.py            # Plotting utilities
```

## Datasets

Experiments on real data expect the following files in `src/datasets/`:

```bash
mkdir -p src/datasets
```

| File | Dataset |
|------|---------|
| `compas.csv` | ProPublica COMPAS |
| `german_credit.csv` | UCI German Credit |
| `law_school.csv` | LSAC Law School |
| `communities.data` | UCI Communities and Crime |
| `communities.names` | Communities and Crime attribute names |

Synthetic experiments (`experiment_approx_ratio.py`, `experiment_scalability*.py`, etc.) do not require any external data.

## Running experiments

All experiment scripts are in `src/` and can be run directly:

```bash
cd src
python experiment_approx_ratio.py
python experiment_scalability_numba.py
python experiment_baselines_real.py
```

Most scripts accept subcommands like `run`, `plot`, or `all` -- run without arguments to see usage.

## Citation

```bibtex
@inproceedings{ferrara2026fair,
  title     = {Fair Bubble Sort: Provably Optimal Fair Ranking with Continuous Sensitive Attributes},
  author    = {Ferrara, Antonio and Vitale, Fabio and Miyauchi, Atsushi and Bonchi, Francesco},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```
