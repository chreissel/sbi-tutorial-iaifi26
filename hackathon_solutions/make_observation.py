"""Write the fiducial observation the configs point at.

Saves `hackathon_solutions/obs_stream.npy`: a single (3, nbins, nbins) GD1
stream image simulated at the full fiducial (`TRUE_VALUES`) parameters. Because
every parameter not being inferred is held at truth, the *same* observation is
the right target for every config here (baseline, progenitor-6D, nuisance).

    python hackathon_solutions/make_observation.py
"""

import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import streams_model as sm  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "obs_stream.npy")


def main(seed=42):
    rng = np.random.default_rng(seed)
    z_full = list(sm.TRUE_VALUES.keys())          # all 16 names -> full truth vector
    z = [sm.TRUE_VALUES[k] for k in z_full]
    img = sm.simulate_image(z, infer_params=z_full, rng=rng).astype(np.float32)
    np.save(OUT, img)
    print(f"saved {OUT}  shape={img.shape}  counts/channel={img.sum((1, 2)).astype(int)}")


if __name__ == "__main__":
    main()
