# Hackathon solutions (private)

**This folder is git-ignored (`.gitignore` → `hackathon_solutions/`) so it never
reaches the public remote.** It exists to prove that every task the notebook and
the README hackathon prompt ask for is actually solvable, and to keep the
reference configs in one place for whoever runs the session.

Everything here assumes the repo root as the working directory and the same
environment the student notebook builds:

```bash
python -m venv env && source env/bin/activate
pip install -r requirements.txt
pip install "diffrax==0.6.1" "git+https://github.com/undark-lab/sstrax.git"
```

## What's here

| File | Proves |
|------|--------|
| `03_hackathon_stellar_streams_solutions.ipynb` | the student notebook with **Exercise 1 filled in and the open-ended baseline worked all the way to a posterior** (config, training, plot) |
| `timing_test.py` | the numpy GD1 conversion matches `sstrax` and kills the vmap re-tracing penalty (the performance fix) |
| `make_observation.py` | writes `obs_stream.npy`, the fiducial observation every config points at |
| `run_baseline.py` | end-to-end **baseline task**: make obs → `falcon` train → posterior figure |
| `config_streams.yml` | filled baseline config — infer `(age, logmsat)` |
| `config_progenitor6d.yml` | **scale-up task** — infer the progenitor's full 6-D phase space |
| `config_nuisance.yml` | **marginalisation task** — infer `(age, logmsat)`, marginalise the 8 tidal-stripping parameters |

## How to reproduce

```bash
# performance fix (correctness + timing)
python hackathon_solutions/timing_test.py

# baseline: trains and writes hackathon_solutions/posterior_baseline.png
python hackathon_solutions/run_baseline.py

# scale-up and nuisance tasks (share the same obs_stream.npy)
python hackathon_solutions/make_observation.py
falcon launch -c hackathon_solutions/config_progenitor6d.yml -o hackathon_solutions/output/prog6d --no-interactive
falcon launch -c hackathon_solutions/config_nuisance.yml    -o hackathon_solutions/output/nuisance --no-interactive
```

## Task → solution map

- **Notebook Exercise 1** (parameters move the data): the one worked warm-up in
  the student notebook — simulate at two ages, the older stream is visibly
  longer in `phi1`.
- **The open-ended baseline** the student notebook poses ("Your hackathon"):
  worked to a posterior in the solutions notebook and in `config_streams.yml` —
  the two `Product` priors are `age ∈ [500, 5000]`, `logmsat ∈ [3.0, 4.5]`, and
  `observed: ./obs_stream.npy`.
- **Baseline inference**: `run_baseline.py` → a real 2-D posterior that pulls
  toward the truth and shows the age–mass degeneracy.
- **Infer more parameters** (README prompt): `config_progenitor6d.yml`. Same
  graph, a 6-D `Product` prior, and the data node given
  `infer_params: [xc, yc, zc, vxc, vyc, vzc]`.
- **Marginalise the nuisances** (README prompt): `config_nuisance.yml`. A second
  node `z_nuis` with a prior and **no `evidence:`** → falcon marginalises it.
  `streams_model.StreamImage` accepts a `nuisance_params` list for exactly this.

## Measured results (build machine: Python 3.11, jax 0.10.2, 4 CPU cores, CPU-only torch)

Full numbers in `RESULTS.txt`. Headlines:

- **Performance fix** — numpy GD1 conversion matches `sstrax` to 5e-5 (float32
  level); `791 ms/sim` (vmap re-tracing) → `0.5 ms/sim` (numpy), ~1700× on the
  conversion step.
- **Baseline** — `(age, logmsat)`, 512 sims / 40 epochs, 14 min on 4 CPU cores:
  `age = 2944 ± 171` (truth 3000), `logmsat = 4.076 ± 0.125` (truth 4.05). See
  `posterior_baseline.png`.
- **Scale-up 6-D** and **nuisance marginalisation** — both train and sample
  (short proof runs), posteriors of shape `(N, 6)` and `(N, 2)` respectively.
