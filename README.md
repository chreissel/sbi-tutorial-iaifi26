# Simulation-Based Inference Tutorial

A hands-on introduction to simulation-based inference (SBI) for the [IAIFI
Summer School 2026](https://iaifi.org/phd-summer-school.html). 

The tutorial is meant to accompany Prof. Christoph Weniger's lectures on the same topic and are therefore heavily based on his existing SBI teaching material
(see [Credits](#credits)).

## The notebooks

The tutorial consists of one notebook each. You should work with the main notebook and check yourself against the matching `_solutions` version.

1. **Simulation-based Inference from Scratch** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chreissel/sbi-tutorial-iaifi26/blob/main/01_sbi_foundations.ipynb)
2. **Real-world example: Gravitational-wave chirp** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chreissel/sbi-tutorial-iaifi26/blob/main/02_gw_falcon.ipynb)
3. **Hackathon: Milky-Way stellar streams** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chreissel/sbi-tutorial-iaifi26/blob/main/03_hackathon_stellar_streams.ipynb)

## Getting set up

1. **Google Colab (recommended).** Click a badge above. The **first cell** of each
notebook clones this repo and installs the requirements — run it first. Then set
**Runtime → Change runtime type → T4 GPU**. Notebook 1 runs top-to-bottom on a
free CPU runtime in **under 5 minutes**; notebook 2 wants the **T4 GPU** for its
`falcon` training runs.

2. **Local machine or cluster.** Use a recent Python (tested with 3.11+) and,
ideally, a CUDA GPU for notebook 2. Then:

```bash
git clone https://github.com/chreissel/sbi-tutorial-iaifi26.git
cd sbi-tutorial-iaifi26
pip install -r requirements.txt
# both notebooks also install Weniger's course package for the simulators:
pip install --no-deps git+https://github.com/cweniger/teaching-2606-ICTP-SAIFR.git
jupyter lab
```

The notebooks are the source of truth — open and run the `.ipynb` files
directly.

## Credits

These tutorials re-cut material from Christoph Weniger's SBI courses: the
simulators, the flow-matching code, the compression and sequential loop, and much
of the prose were taken from the following sources:

- **ICTP-SAIFR** — [`cweniger/teaching-2606-ICTP-SAIFR`](https://github.com/cweniger/teaching-2606-ICTP-SAIFR)
  (the ball-throw and gravitational-wave simulators, the heteroscedastic Gaussian
  head, the matched-filter summary).
- **LISA Hackathon** — [`cweniger/teaching-2607-LISA-Hackathon`](https://github.com/cweniger/teaching-2607-LISA-Hackathon)
  (the ten-line flow-matching loss, data compression, and the sequential/dynamic
  SBI loop).
- **`falcon`** — [`cweniger/falcon`](https://github.com/cweniger/falcon), the
  CLI-driven SBI framework used in notebooks 2 and 3.
- **stellar streams** — the notebook-3 simulator and binning are ported from
  [`sstrax`](https://github.com/undark-lab/sstrax) and
  [`albatross`](https://github.com/undark-lab/albatross) (Alvey, Gerdes &
  Weniger, [arXiv:2304.02032](https://arxiv.org/abs/2304.02032)), a swyft/TMNRE
  GD1-stream pipeline; here re-expressed as a `falcon` graph.

The production-scale dynamic-SBI results referenced at the end of notebook 2 are
from Alvey, Lyu, Weniger et al., [arXiv:2510.13997](https://arxiv.org/abs/2510.13997).

## Hackathon prompt

Stellar streams are among the best dynamical probes we have of the Milky Way's gravitational potential and its dark matter. A globular cluster or dwarf galaxy caught in the Galactic tide slowly unravels into a thin ribbon of stars strung along its orbit, and the ribbon's length, width, and kinematics encode both the progenitor that made it and the potential it fell through. That very sensitivity is what makes them hard to analyse: the forward model is a stochastic, multi-Gyr disruption simulation with **no tractable likelihood** — each run strips a *different, variable* number of stars at random moments, and the stream reaches us as a noisy, field-star-contaminated point cloud with uncertain membership. There is simply nothing to write `p(data | θ)` down for.

That is exactly the regime simulation-based inference was built for: a simulator you can sample but not evaluate, sixteen parameters mixing what you care about (the progenitor's mass, age, and orbit) with tidal-stripping nuisances you don't, and variable-length data that has to be *learned* into a fixed-size summary before any network can read it. So: use `falcon` to infer the GD1 progenitor's parameters from a binned image of its stream, starting from the age–mass baseline in notebook 3 and scaling up — toward the progenitor's full phase space, marginalising the eight stripping nuisances into a node with no `evidence:`, and checking whether the posteriors are actually calibrated (the notebook-1 coverage tests). Compare everything honestly on **simulation budget**, the number of simulator calls. The simulator is [`sstrax`](https://github.com/undark-lab/sstrax), a `jax` model of GD1; a small 2-D CNN over the three (sky, proper-motion, distance–velocity) channels makes a good data embedding; and the production reference is [`albatross`](https://github.com/undark-lab/albatross) (Alvey, Gerdes & Weniger, [arXiv:2304.02032](https://arxiv.org/abs/2304.02032)), the swyft pipeline this problem is ported from.
