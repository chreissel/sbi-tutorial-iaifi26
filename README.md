# Simulation-Based Inference — a two-session tutorial

A hands-on introduction to **simulation-based inference (SBI)** for the IAIFI
Summer School. Two notebooks, ~60 minutes each, built to run on Google Colab.

The tutorials are a re-cut of Christoph Weniger's existing SBI teaching material
(see [Credits](#credits)); most of the code is his, reorganised into one
continuous arc from a linear-regression posterior to a gravitational-wave
inference run driven by [`falcon`](https://github.com/cweniger/falcon).

## Audience and prerequisites

For students with a solid general deep-learning background who are **new to SBI**.
We assume you are already comfortable with:

- PyTorch tensors, `autograd`, and `nn.Module`;
- writing and reading a training loop; learning rates, overfitting, early stopping.

We do **not** re-teach any of that. If those are unfamiliar, work through the
first hour of Weniger's LISA-Hackathon tutorial (linked below) first.

## The notebooks

Each tutorial comes as a **pair of files**: a `_student` notebook with the
exercise cells left as `# TODO`, and a `_solutions` notebook with the reference
answers filled in. The two are identical outside the exercise cells. Work through
the student version; check yourself against the solutions.

| Notebook | Summary | Student | Solutions |
|---|---|---|---|
| **1 — SBI from scratch** | From a Gaussian regression head to a calibrated posterior: NPE, why the loss returns the posterior, where Gaussians break, mixture + flow-matching density estimators, simulation-based calibration, and amortised vs sequential inference. Pure PyTorch, CPU-friendly. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chreissel/sbi-tutorial-iaifi26/blob/main/01_sbi_foundations_student.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chreissel/sbi-tutorial-iaifi26/blob/main/01_sbi_foundations_solutions.ipynb) |
| **2 — A GW chirp with `falcon`** | The same ideas on a realistic gravitational-wave chirp: whitening, matched-filter compression, then the whole inference declared as a YAML graph in `falcon` — a fixed summary vs a learned CNN embedding, adaptivity, and a nuisance node. | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chreissel/sbi-tutorial-iaifi26/blob/main/02_gw_falcon_student.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chreissel/sbi-tutorial-iaifi26/blob/main/02_gw_falcon_solutions.ipynb) |

## Getting set up

**Google Colab (recommended).** Click a badge above. The **first cell** of each
notebook clones this repo and installs the requirements — run it first. Then set
**Runtime → Change runtime type → T4 GPU**. Notebook 1 runs top-to-bottom on a
free CPU runtime in **under 5 minutes**; notebook 2 wants the **T4 GPU** for its
`falcon` training runs.

> **Forked or renamed the repo?** The Colab badges, the `git clone` in the first
> cell, and the Release-asset URL in notebook 2 all hard-code
> `chreissel/sbi-tutorial-iaifi26`. Change that slug to your own `USER/REPO`
> everywhere it appears (each notebook flags the spots inline).

**Local machine or cluster.** Use a recent Python (tested with 3.11+) and,
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

## The pre-computed `falcon` run (notebook 2 §4)

Training the learned-CNN embedding to convergence takes far longer than a
session allows. Notebook 2 therefore runs the CNN **live but small** (so you can
watch it being deliberately under-budget), then downloads a **pre-computed
converged run** from a GitHub **Release asset** and overlays it. This mirrors how
the LISA-Hackathon repo distributes its `mbhb_simbank.npz`.

The download URL in the notebook is a **placeholder**:

```
https://github.com/chreissel/sbi-tutorial-iaifi26/releases/download/v0.1-artifacts/gw_cnn_converged.npz
```

To make it live, run notebook 2 with a large buffer and many epochs to
convergence, save the posterior samples as `gw_cnn_converged.npz` (an array under
key `z`), and upload it as an asset on a GitHub Release named `v0.1-artifacts`
(replacing `chreissel/sbi-tutorial-iaifi26` with your own repo if you forked it).
The artifact is intentionally **not committed** to the repository. If the
download fails, the notebook falls back to the live CNN run so everything still
executes end-to-end.

## Credits

These tutorials re-cut material from Christoph Weniger's SBI courses — the
simulators, the flow-matching code, the compression and sequential loop, and much
of the prose are his:

- **ICTP-SAIFR** — [`cweniger/teaching-2606-ICTP-SAIFR`](https://github.com/cweniger/teaching-2606-ICTP-SAIFR)
  (the ball-throw and gravitational-wave simulators, the heteroscedastic Gaussian
  head, the matched-filter summary).
- **LISA Hackathon** — [`cweniger/teaching-2607-LISA-Hackathon`](https://github.com/cweniger/teaching-2607-LISA-Hackathon)
  (the ten-line flow-matching loss, data compression, and the sequential/dynamic
  SBI loop).
- **`falcon`** — [`cweniger/falcon`](https://github.com/cweniger/falcon), the
  CLI-driven SBI framework used in notebook 2.

The production-scale dynamic-SBI results referenced at the end of notebook 2 are
from Alvey, Lyu, Weniger et al., [arXiv:2510.13997](https://arxiv.org/abs/2510.13997).

The repository layout follows Sam Bright-Thonney's
[manifold-capacity tutorial](https://github.com/sambt/manifold-capacity-tutorial-iaifi25)
from the 2025 IAIFI Summer School.
