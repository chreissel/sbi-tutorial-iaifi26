"""Milky Way stellar-stream simulator, wrapped for `falcon`.

This is the forward model for notebook 3 (the IAIFI hackathon). It ports the
GD1-stream pipeline from `albatross` (undark-lab/albatross, a swyft/TMNRE
pipeline) onto Christoph Weniger's `falcon` framework, keeping the same
simulator numbers.

The physics lives in `sstrax` (undark-lab/sstrax): given 16 parameters it
integrates a disrupting star cluster and returns the phase-space coordinates of
the tidal-stream stars in the Milky-Way (`halo`) frame. Here we

  1. run `sstrax.simulate_stream`  ->  (N_stars, 6) halo-frame phase space,
  2. rotate into observable GD1 stream coordinates (`stars_to_gd1`),
  3. add per-observable Gaussian errors and drop a few stars (`add_noise`),
  4. add a uniform foreground of Milky-Way field stars (`sample_background`),
  5. bin everything into three fixed-shape 2-D histograms (`bin_stream`),

so that every parameter vector maps to a fixed `(3, nbins, nbins)` image — the
fixed shape is what lets a single CNN read the data no matter how many stars a
particular stream happens to have.

--------------------------------------------------------------------------------
Two things worth knowing if you read `albatross` alongside this file
--------------------------------------------------------------------------------

* `sstrax.simulate_stream(...)` returns shape **(N_stars, 6)** = (x, y, z, vx,
  vy, vz). The sstrax README says (6, N_stars); it is wrong. albatross indexes
  it as (N_stars, 6) and so do we.

* albatross converts to GD1 with `sstrax.halo_to_gd1_vmap` /
  `..._velocity_vmap`. Those are *jitted* jax vmaps, and because `N_stars`
  changes on **every** simulation they re-trace (re-compile) every single call —
  ~0.7 s of pure compilation per sim, several times the cost of the simulation
  itself. We reimplement the exact same transform in plain numpy below
  (`stars_to_gd1`). It is a fixed rotation + unit conversions, so numpy
  reproduces sstrax to float precision while costing ~0.5 ms instead of ~700 ms.
  See `hackathon_solutions/` for the timing test.

Falcon imports this module inside its Ray workers (see `paths.imports` in the
config), so everything here must be importable with no side effects beyond the
one-off constants below.
"""

import numpy as np
import jax

# The physics package. Importing sstrax pulls in jax. `PRIOR_LIST` and
# `Parameters` come from sstrax.constants.
import sstrax
from sstrax.constants import PRIOR_LIST  # noqa: F401  (re-exported for convenience)


# =============================================================================
# Simulator numbers — lifted verbatim from albatross' example config
# (undark-lab/albatross:examples/configs/example_config.txt). Do not invent
# new ones; these are the ranges/errors the pipeline was designed around.
# =============================================================================

# The 16 sstrax parameters, in sstrax order:
#   xc yc zc vxc vyc vzc age msat xi0 alpha rh mbar sigv lrelease lmatch stripnear
# Prior ranges. Mass is handled in log10 (name "logmsat") the way albatross does
# it, because msat spans decades; everything else is sampled linearly.
PRIOR_RANGES = {
    "xc": (10.0, 14.0),
    "yc": (0.1, 2.5),
    "zc": (6.0, 8.0),
    "vxc": (90.0, 115.0),
    "vyc": (-280.0, -230.0),
    "vzc": (-120.0, -80.0),
    "age": (500.0, 5000.0),
    "logmsat": (3.0, 4.5),         # msat = 10**logmsat  (Msun)
    "xi0": (0.0001, 0.01),
    "alpha": (10.0, 30.0),
    "rh": (0.0001, 0.01),
    "mbar": (1.0, 20.0),
    "sigv": (0.1, 5.0),
    "lrelease": (0.1, 2.0),
    "lmatch": (0.1, 2.0),
    "stripnear": (0.0, 1.0),
}

# Fiducial ("true") parameters that define the observation and fill in whatever
# is not being inferred.
TRUE_VALUES = {
    "xc": 11.8, "yc": 0.79, "zc": 6.4,
    "vxc": 109.5, "vyc": -254.5, "vzc": -90.3,
    "age": 3000.0, "logmsat": 4.05,
    "xi0": 0.001, "alpha": 20.9, "rh": 0.001, "mbar": 3.0,
    "sigv": 1.1, "lrelease": 1.405, "lmatch": 1.846, "stripnear": 0.5,
}

# Histogram ranges for the three observable channels.
BINNING = {
    "phi1": (-120.0, 70.0),
    "phi2": (-8.0, 2.0),
    "pm_phi1_cosphi2": (-2.0, 1.0),
    "pm_phi2": (-0.1, 0.1),
    "vrad": (-250.0, 250.0),
    "dist": (6.0, 20.0),
}

# Per-observable Gaussian errors, selection efficiency, and the field-star
# background (2e6 * 1e-5 = 20 uniformly-scattered contaminating stars).
ERRORS = {
    "phi1": 0.001,
    "phi2": 0.15,
    "pm_phi1_cosphi2": 0.1,
    "pm_phi2": 0.0,
    "vrad": 5.0,
    "dist": 0.25,
    "stream_selection": 0.95,      # keep this fraction of stream stars
    "total_background": 2e6,
    "background_removal": 1e-5,
}

# Default image resolution. albatross used a non-square (64, 32); we use one
# square number so the three channels stack into a clean (3, NBINS, NBINS)
# tensor for the CNN. Smaller = faster training, coarser images.
NBINS = 48

# The small, intuitive parameter block notebook 3 infers by default: the
# stream's disruption age and its progenitor mass. Both visibly reshape the
# stream (older / heavier -> longer, denser), so the 2-D posterior is easy to
# read against the truth. The hackathon is about growing this list.
DEFAULT_INFER = ["age", "logmsat"]


# =============================================================================
# 1. sstrax parameters  <->  inference vector
# =============================================================================

def _key_from(rng):
    """A fresh jax PRNGKey seeded from a numpy Generator.

    sstrax needs a jax key per simulation; albatross draws it from a random int,
    which is exactly what we do so that falcon's parallel workers decorrelate.
    """
    rng = np.random.default_rng() if rng is None else rng
    return jax.random.PRNGKey(int(rng.integers(0, 2**31 - 1)))


def params_from_vector(z, infer_params):
    """Build an sstrax `Parameters` from an inference vector `z`.

    `z[i]` is the value of `infer_params[i]`; every other parameter is held at
    its fiducial `TRUE_VALUES`. The name "logmsat" maps onto `msat = 10**value`.
    """
    z = np.asarray(z, dtype=float).ravel()
    vals = dict(TRUE_VALUES)
    for name, v in zip(infer_params, z):
        vals[name] = float(v)
    kwargs = {}
    for name in PRIOR_LIST:                       # sstrax field names
        if name == "msat":
            kwargs["msat"] = 10.0 ** vals["logmsat"]
        else:
            kwargs[name] = vals[name]
    return sstrax.Parameters(**kwargs)


# =============================================================================
# 2. halo -> GD1, in plain numpy (the performance fix)
# =============================================================================
# Position: exact numpy port of the sstrax.projection chain
#   halo_to_sun -> sun_to_gal -> gal_to_equat -> equat_to_gd1cart -> gd1cart_to_gd1
# The whole halo -> GD1-cartesian map is affine (a rotation + a translation), so
# its Jacobian is a single constant 3x3 matrix `_C` that we extract once. The
# velocity transform is then that constant matrix followed by the analytic
# Jacobian of the (r, phi1, phi2) spherical map — no jax, no re-tracing.

def _halo_to_gd1cart(Xhalo):
    """(N, 3) halo-frame positions -> (N, 3) cartesian GD1 positions."""
    x, y, z = Xhalo[:, 0], Xhalo[:, 1], Xhalo[:, 2]
    xsun, ysun, zsun = 8.0 - x, y, z                       # halo_to_sun
    r = np.sqrt(xsun ** 2 + ysun ** 2 + zsun ** 2)         # sun_to_gal
    b = np.arcsin(zsun / r)
    l = np.arctan2(ysun, xsun)
    # gal_to_equat (Galactic -> equatorial rotation via NGP constants)
    dNGP = 27.12825118085622 * np.pi / 180.0
    lNGP = 122.9319185680026 * np.pi / 180.0
    aNGP = 192.85948 * np.pi / 180.0
    sb, cb = np.sin(b), np.cos(b)
    sl, cl = np.sin(lNGP - l), np.cos(lNGP - l)
    alpha = np.arctan((cb * sl) / (np.cos(dNGP) * sb - np.sin(dNGP) * cb * cl)) + aNGP
    delta = np.arcsin(np.sin(dNGP) * sb + np.cos(dNGP) * cb * cl)
    ca, sa, cd, sd = np.cos(alpha), np.sin(alpha), np.cos(delta), np.sin(delta)
    # equat_to_gd1cart (fixed rotation matrix)
    xg = r * (-0.4776303088 * ca * cd - 0.1738432154 * sa * cd + 0.8611897727 * sd)
    yg = r * (0.510844589 * ca * cd - 0.8524449229 * sa * cd + 0.111245042 * sd)
    zg = r * (0.7147776536 * ca * cd + 0.4930681392 * sa * cd + 0.4959603976 * sd)
    return np.stack([xg, yg, zg], axis=1)


# Constant Jacobian of the affine halo -> GD1-cartesian map (extracted once).
_ORIGIN = _halo_to_gd1cart(np.zeros((1, 3)))[0]
_C = np.stack(
    [_halo_to_gd1cart(np.eye(3)[j:j + 1])[0] - _ORIGIN for j in range(3)], axis=1
)  # _C[:, j] = d(gd1cart)/d(halo_j)

# Unit conversions (identical to albatross' stars_to_gd1)
_KPCMYR_TO_KMS = 977.7922216807891         # kpc/Myr -> km/s
_RADMYR_TO_MASYR = 2.0626480624709636e8 / 1e6  # rad/Myr -> mas/yr


def stars_to_gd1(stars):
    """(N, 6) halo phase space -> (N, 6) GD1 observables.

    Columns of the output, matching albatross:
      0 dist  [kpc]        3 vrad             [km/s]
      1 phi1  [deg]        4 pm_phi1_cosphi2  [mas/yr]
      2 phi2  [deg]        5 pm_phi2          [mas/yr]
    """
    stars = np.asarray(stars, dtype=float)
    Xh, Vh = stars[:, :3], stars[:, 3:]
    Xc = _halo_to_gd1cart(Xh)          # cartesian GD1 positions
    Vc = Vh @ _C.T                     # cartesian GD1 velocities (affine -> linear)
    x, y, z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    rho2 = x ** 2 + y ** 2
    rho = np.sqrt(rho2)
    phi1 = np.arctan2(y, x)
    phi2 = np.arcsin(z / r)
    vx, vy, vz = Vc[:, 0], Vc[:, 1], Vc[:, 2]
    # analytic Jacobian of (r, phi1, phi2) w.r.t. cartesian, applied to velocity
    dr = (x * vx + y * vy + z * vz) / r
    dphi1 = (-y * vx + x * vy) / rho2
    dphi2 = (-x * z * vx - y * z * vy + rho2 * vz) / (r ** 2 * rho)

    gd1 = np.empty((len(stars), 6))
    gd1[:, 0] = r                                   # dist  [kpc]
    gd1[:, 1] = phi1 * 180.0 / np.pi                # phi1  [deg]
    gd1[:, 2] = phi2 * 180.0 / np.pi                # phi2  [deg]
    gd1[:, 3] = dr * _KPCMYR_TO_KMS                 # vrad  [km/s]
    gd1[:, 4] = dphi1 / r * _RADMYR_TO_MASYR        # pm_phi1_cosphi2 [mas/yr]
    gd1[:, 5] = dphi2 / r * _RADMYR_TO_MASYR        # pm_phi2 [mas/yr]
    return gd1


# =============================================================================
# 3. noise, background, binning  (lifted from albatross, largely verbatim)
# =============================================================================

def add_noise(gd1, rng=None):
    """Add per-observable Gaussian errors and keep a random `stream_selection`
    fraction of the stars. `gd1` columns are (dist, phi1, phi2, vrad,
    pm_phi1_cosphi2, pm_phi2)."""
    rng = np.random.default_rng() if rng is None else rng
    gd1 = np.array(gd1, dtype=float)
    n = len(gd1)
    gd1[:, 0] += rng.normal(0.0, ERRORS["dist"], n)
    gd1[:, 1] += rng.normal(0.0, ERRORS["phi1"], n)
    gd1[:, 2] += rng.normal(0.0, ERRORS["phi2"], n)
    gd1[:, 3] += rng.normal(0.0, ERRORS["vrad"], n)
    gd1[:, 4] += rng.normal(0.0, ERRORS["pm_phi1_cosphi2"], n)
    gd1[:, 5] += rng.normal(0.0, ERRORS["pm_phi2"], n)
    keep = int(np.floor(ERRORS["stream_selection"] * n))
    idx = rng.choice(n, size=keep, replace=False)
    return gd1[idx, :]


def sample_background(rng=None):
    """Uniform Milky-Way field-star contamination in the observable windows.

    `total_background * background_removal` stars, scattered uniformly across the
    binning ranges — an irreducible, parameter-independent floor under the
    stream."""
    rng = np.random.default_rng() if rng is None else rng
    num = int(np.floor(ERRORS["total_background"] * ERRORS["background_removal"]))
    bg = np.zeros((num, 6))
    bg[:, 0] = rng.uniform(*BINNING["dist"], num)
    bg[:, 1] = rng.uniform(*BINNING["phi1"], num)
    bg[:, 2] = rng.uniform(*BINNING["phi2"], num)
    bg[:, 3] = rng.uniform(*BINNING["vrad"], num)
    bg[:, 4] = rng.uniform(*BINNING["pm_phi1_cosphi2"], num)
    bg[:, 5] = rng.uniform(*BINNING["pm_phi2"], num)
    return bg


def bin_stream(gd1, nbins=NBINS):
    """(N, 6) GD1 stars -> (3, nbins, nbins) stack of 2-D count histograms:
      channel 0: (phi1, phi2)                  — the stream on the sky
      channel 1: (pm_phi1_cosphi2, pm_phi2)    — proper motions
      channel 2: (dist, vrad)                  — distance vs radial velocity
    """
    dist, phi1, phi2, vrad, pm1, pm2 = (gd1[:, i] for i in range(6))
    sky, _, _ = np.histogram2d(phi1, phi2, bins=nbins,
                               range=[BINNING["phi1"], BINNING["phi2"]])
    pm, _, _ = np.histogram2d(pm1, pm2, bins=nbins,
                              range=[BINNING["pm_phi1_cosphi2"], BINNING["pm_phi2"]])
    dv, _, _ = np.histogram2d(dist, vrad, bins=nbins,
                              range=[BINNING["dist"], BINNING["vrad"]])
    return np.stack([sky, pm, dv]).astype(np.float32)


# =============================================================================
# 4. one full forward simulation
# =============================================================================

def simulate_image(z, infer_params=DEFAULT_INFER, nbins=NBINS,
                   with_background=True, key=None, rng=None):
    """Full forward model: inference vector `z` -> (3, nbins, nbins) image.

    Runs sstrax, rotates to GD1, adds noise, adds the field-star background, and
    bins. `key` (a jax PRNGKey) seeds the stochastic stream; `rng` (a numpy
    Generator) seeds the observational noise and background.
    """
    rng = np.random.default_rng() if rng is None else rng
    key = _key_from(rng) if key is None else key
    params = params_from_vector(z, infer_params)
    stars = np.asarray(sstrax.simulate_stream(key=key, params=params))  # (N, 6)
    gd1 = add_noise(stars_to_gd1(stars), rng=rng)
    image = bin_stream(gd1, nbins=nbins)
    if with_background:
        image = image + bin_stream(sample_background(rng=rng), nbins=nbins)
    return image


# =============================================================================
# 5. falcon simulator node
# =============================================================================

class StreamImage:
    """Falcon data node: parameters -> a `(3, nbins, nbins)` GD1 image.

    A falcon simulator node is any object with a
    `simulate_batch(self, batch_size, *parents)` method returning a numpy array
    with a leading batch axis. The first parent, `z`, is the inference vector
    (the subset of parameters named by `infer_params`, in that order). An
    optional second parent, `z_nuis`, carries *nuisance* parameters named by
    `nuisance_params`: give that node a prior but no `evidence:` in the config,
    and falcon samples it and marginalises over it automatically (see notebook
    2's nuisance example). Everything not in either list stays at its fiducial
    value.
    """

    def __init__(self, infer_params=DEFAULT_INFER, nuisance_params=None,
                 nbins=NBINS, with_background=True):
        self.infer_params = list(infer_params)
        self.nuisance_params = list(nuisance_params) if nuisance_params else []
        self.nbins = int(nbins)
        self.with_background = bool(with_background)

    def simulate_batch(self, batch_size, z, z_nuis=None):
        names = self.infer_params + self.nuisance_params
        z = np.asarray(z, dtype=float).reshape(-1, len(self.infer_params))
        if self.nuisance_params:
            z_nuis = np.asarray(z_nuis, dtype=float).reshape(-1, len(self.nuisance_params))
            Z = np.concatenate([z, z_nuis], axis=1)
        else:
            Z = z
        out = np.empty((len(Z), 3, self.nbins, self.nbins), dtype=np.float32)
        for i, zi in enumerate(Z):
            out[i] = simulate_image(
                zi, infer_params=names, nbins=self.nbins,
                with_background=self.with_background,
            )
        return out


# =============================================================================
# 6. a data embedding for the falcon Flow estimator
# =============================================================================

import torch.nn as nn  # noqa: E402  (falcon always provides torch)


class StreamCNN(nn.Module):
    """Small 2-D CNN: (B, 3, nbins, nbins) stream image -> (B, out_features).

    This is the data compression falcon's Flow sits on top of: it turns the
    three count-histograms into `out_features` numbers. The channels are integer
    counts of very different magnitudes, so we normalize each image to
    zero-mean / unit-variance before the convolutions — the same trick notebook
    2's CNN used on raw strain, so gradients do not vanish.
    """

    def __init__(self, out_features=16, channels=(16, 32, 64)):
        super().__init__()
        c1, c2, c3 = channels
        self.net = nn.Sequential(
            nn.Conv2d(3, c1, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(c1, c2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(c2, c3, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(2), nn.Flatten(),
            nn.Linear(c3 * 2 * 2, out_features),
        )

    def forward(self, x):
        x = x.float()
        # normalize per image so the very different channel count scales do not
        # swamp the gradients
        mu = x.mean(dim=(-1, -2, -3), keepdim=True)
        sd = x.std(dim=(-1, -2, -3), keepdim=True)
        return self.net((x - mu) / (sd + 1e-8))
