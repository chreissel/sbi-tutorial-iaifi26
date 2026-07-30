"""End-to-end baseline: make the observation, train with falcon, load the
posterior, and save a figure — the proof that the notebook-3 baseline task
runs to a real answer.

    python hackathon_solutions/run_baseline.py

Writes:
    hackathon_solutions/obs_stream.npy
    hackathon_solutions/output/baseline/...        (falcon run + posterior samples)
    hackathon_solutions/posterior_baseline.png     (posterior vs truth)
"""

import glob
import os
import subprocess
import sys
import warnings

import numpy as np
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import streams_model as sm          # noqa: E402
import streams_plotting as sp       # noqa: E402
import make_observation             # noqa: E402

CONFIG = os.path.join(HERE, "config_streams.yml")
RUNDIR = os.path.join(HERE, "output", "baseline")
INFER = ["age", "logmsat"]


def load_posterior(run_dir):
    files = sorted(glob.glob(f"{run_dir}/samples/posterior/*.npz"))
    return np.concatenate([np.atleast_2d(np.load(f)["z"]) for f in files], axis=0)


def main():
    make_observation.main()
    print("training with falcon (this simulates + fits; ~10 min on CPU) ...")
    subprocess.run(
        ["falcon", "launch", "-c", CONFIG, "-o", RUNDIR, "--no-interactive"],
        cwd=ROOT, check=True,
    )
    post = load_posterior(RUNDIR)
    truth = [sm.TRUE_VALUES[p] for p in INFER]
    print(f"posterior shape {post.shape}")
    for i, p in enumerate(INFER):
        print(f"  {p:8s}: {post[:, i].mean():9.3f} +/- {post[:, i].std():7.3f}"
              f"   (truth {truth[i]})")
    fig = sp.plot_posterior(post, INFER, truth=truth,
                            title="baseline falcon posterior vs truth")
    out = os.path.join(HERE, "posterior_baseline.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
