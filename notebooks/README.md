# Notebooks

Analysis notebooks for `System0Sandbox` runs. Install the extra deps with
`pip install -e ".[notebooks]"`.

- **`telemetry_analysis.ipynb`** — Loads a single saved run (`results/<config>/<run_id>/`)
  via `metabolic_intelligence_lab.persistence` and plots reward/CES-like efficiency curves,
  salience-vs-threshold arbitration, energy/survival pressure, prospective-forecast accuracy
  (from `ProspectiveLog`), and goal-frontier composition. Run `python experiments/run_demo.py`
  first to generate a run to inspect.

- **`phenotype_comparison.ipynb`** — Runs the configs in `configs/*.yaml` head-to-head through
  `System0Sandbox` and compares CES-like efficiency, survival rate, energy spend, salience
  entropy/stability, and prospective-forecast accuracy across phenotypes (e.g. energy-conservative
  vs. exploratory vs. high-prospective-depth).
