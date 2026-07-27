# Simulation-Based Inference Tutorial

A hands-on introduction to simulation-based inference (SBI) for the [IAIFI
Summer School 2026](https://iaifi.org/phd-summer-school.html). 

The tutorial is meant to accompany Prof. Christoph Weniger's lectures on the same topic and are therefore heavily based on his existing SBI teaching material
(see [Credits](#credits)).

## The notebooks

The tutorial consists of one notebook each. You should work with the notebook labeled `student`, and check yourself against the solutions.

1. **Simulation-based Inference from Scratch** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chreissel/sbi-tutorial-iaifi26/blob/main/01_sbi_foundations_student.ipynb)
2. **Real-world example: Gravitational-wave chirp** [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/chreissel/sbi-tutorial-iaifi26/blob/main/02_gw_falcon_student.ipynb)

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
  CLI-driven SBI framework used in notebook 2.

The production-scale dynamic-SBI results referenced at the end of notebook 2 are
from Alvey, Lyu, Weniger et al., [arXiv:2510.13997](https://arxiv.org/abs/2510.13997).
