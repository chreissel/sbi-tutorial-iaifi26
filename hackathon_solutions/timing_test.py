"""Proof of the performance fix that makes the hackathon tractable.

albatross converts halo -> GD1 coordinates with the jitted jax vmaps
`sstrax.halo_to_gd1_vmap` / `sstrax.halo_to_gd1_velocity_vmap`. Because the
number of stream stars changes on *every* simulation, those vmaps re-trace
(re-compile) each call — hundreds of ms of pure compilation per sim. The numpy
reimplementation in `streams_model.stars_to_gd1` reproduces the same transform
to float precision at ~0.5 ms.

Run from the repo root inside the streams venv:
    python hackathon_solutions/timing_test.py

Measured on the build machine (Python 3.11, jax 0.10.2, 4 CPU cores):
    numpy conversion matches sstrax to < 1e-4 abs error (float32 level)
    sstrax vmap conversion : ~730 ms / sim   (re-tracing on every new N)
    numpy  conversion      : ~0.5 ms / sim   -> ~1400x on the conversion step
so the numpy version turns a >3 s/sim chain back into a ~0.5 s/sim one.
"""

import os
import sys
import time
import warnings

import numpy as np
import jax

warnings.filterwarnings("ignore")

# import streams_model from the repo root regardless of where we're called from
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import sstrax  # noqa: E402
import streams_model as sm  # noqa: E402


def sstrax_vmap_conversion(stars):
    """albatross' original conversion: jitted vmaps + the same unit factors."""
    Xhalo, Vhalo = stars[:, :3], stars[:, 3:]
    Xgd1 = np.array(sstrax.halo_to_gd1_vmap(Xhalo))
    Vgd1 = np.array(sstrax.halo_to_gd1_velocity_vmap(Xhalo, Vhalo))
    Vgd1[:, 0] = Vgd1[:, 0] * 977.7922216807891
    Vgd1[:, 1] = Vgd1[:, 1] / Xgd1[:, 0] * 2.0626480624709636e8 / 1e6
    Vgd1[:, 2] = Vgd1[:, 2] / Xgd1[:, 0] * 2.0626480624709636e8 / 1e6
    Xgd1[:, 1] = Xgd1[:, 1] * 180.0 / np.pi
    Xgd1[:, 2] = Xgd1[:, 2] * 180.0 / np.pi
    return np.concatenate((Xgd1, Vgd1), axis=1)


def main():
    # --- correctness: numpy vs sstrax on one stream ---
    stars = np.asarray(sstrax.simulate_stream(
        key=jax.random.PRNGKey(0), params=sstrax.Parameters()))
    print(f"simulate_stream shape: {stars.shape}  (expect (N, 6))")
    ref = sstrax_vmap_conversion(stars)
    mine = sm.stars_to_gd1(stars)
    print(f"max abs error numpy vs sstrax: {np.abs(ref - mine).max():.2e}  "
          "(float32-level -> numpy float64 is if anything cleaner)")

    # --- timing: fresh N on every call (the real simulation-loop scenario) ---
    ages = np.linspace(700, 3200, 12)
    starslist = [np.asarray(sstrax.simulate_stream(
        key=jax.random.PRNGKey(300 + i), params=sstrax.Parameters(age=float(a))))
        for i, a in enumerate(ages)]
    print("N_stars over the age sweep:", sorted(len(s) for s in starslist))

    sstrax_vmap_conversion(starslist[0])   # warm the jit
    sm.stars_to_gd1(starslist[0])

    t0 = time.time()
    for s in starslist:
        sstrax_vmap_conversion(s)
    t_ref = (time.time() - t0) / len(starslist)

    t0 = time.time()
    for s in starslist:
        sm.stars_to_gd1(s)
    t_np = (time.time() - t0) / len(starslist)

    print(f"sstrax vmap conversion : {t_ref * 1e3:8.1f} ms/sim")
    print(f"numpy  conversion      : {t_np * 1e3:8.1f} ms/sim")
    print(f"speedup on conversion  : {t_ref / t_np:8.0f}x")


if __name__ == "__main__":
    main()
