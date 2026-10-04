# ai-debugging-simulation

Monte Carlo study of the operating characteristics of an analysis plan for a two-group
experiment on AI assistants and code debugging (N = 40). **All data are synthetic.**

## Contents

- `sim_core.py` — data generator (parameters are the author's assumptions)
- `analysis_core.py` — confirmatory analyses (Cox, Wilcoxon, ANCOVA, TOST, Holm)
- `scenarios_simulation.py` — scenarios S0–S5, variants v1/v2/stratified, paired TOST
- `run_demo.py` — one illustrative dataset (seed 2026) with report and figures
- `output/` — results (raw replications, summaries, truth values, figures)

## Requirements

Python 3.12; packages pinned in `requirements.txt`.

## Reproduce

    python -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    python scenarios_simulation.py --selftest
    python scenarios_simulation.py --n-sim 2000 --workers 4 --only S0_null S1_planted S3_viol6 S5_boundary
    python scenarios_simulation.py --n-sim 1000 --workers 4 --only S2_N60 S2_N80 S2_N100 S2_N120 S2_N160
    python scenarios_simulation.py --from-raw

## Licenses

Code: MIT. Data and figures in `output/`: CC BY 4.0.

## Citation

See `CITATION.cff`. Author: Aidos Kozhagul, Kazakh National Women's Pedagogical University.
