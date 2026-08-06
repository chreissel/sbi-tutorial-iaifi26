"""Milky Way stellar-stream simulator, wrapped for `falcon` — pure jax.

This is the forward model for notebook 3 (the IAIFI hackathon). It ports the
GD1-stream pipeline from `albatross` (undark-lab/albatross, a swyft/TMNRE
pipeline) onto Christoph Weniger's `falcon` framework, keeping the same
simulator numbers.

The physics lives in `sstrax` (undark-lab/sstrax): given 16 parameters it
integrates a disrupting star cluster and returns the phase-space coordinates of
the tidal-stream stars in the Milky-Way (`halo`) frame. Here we

  1. run the stripped-star evolution                       ->  (N_stars, 6) halo,
  2. rotate into observable GD1 stream coordinates (`stars_to_gd1`),
  3. add per-observable Gaussian errors and a selection cut (`_add_noise`),
  4. add a uniform foreground of Milky-Way field stars (`_sample_background`),
  5. bin everything into three fixed-shape 2-D histograms (`bin_stream`),

so that every parameter vector maps to a fixed `(3, nbins, nbins)` image — the
fixed shape is what lets a single CNN read the data no matter how many stars a
particular stream happens to have.

--------------------------------------------------------------------------------
Everything runs in jax (and on a GPU). Why that took a trick.
--------------------------------------------------------------------------------

The natural worry with jax here is re-compilation. `sstrax.simulate_stream`
returns a **variable** number of stars (older / heavier progenitors shed more),
and jax re-traces a jitted function whenever an input's shape changes. albatross
hit this two ways: it evolved the stars in a Python `for` loop (one jitted ODE
solve per star, copied to host each iteration — the real cost, ~1.1 s/sim), and
it rotated to GD1 with jitted `vmap`s that re-compiled (~0.7 s) every call
because `N_stars` moved. An earlier version of this file dodged the second cost
by doing the rotation in numpy.

The clean fix removes the root cause instead of working around it: **pin the
star array to a constant `N_MAX` and carry a 0/1 weight mask** marking the truly
present stars. The only thing that changed shape was the star count, so once it
is constant

  * the per-star evolution becomes a single `jax.vmap` that compiles **once** and
    is reused for every parameter vector (no re-trace as the true count changes),
  * the histograms weight by the mask, so the star-count signal is preserved,
  * the whole pipeline (evolve -> GD1 -> noise -> background -> binning) is one
    fixed-shape jax program that runs on whatever device jax sees — including a
    GPU — and `vmap`s over a whole batch of parameter vectors.

So there is no numpy simulator any more: `simulate_image` is the jitted jax model
(`simulate_image_jax` is the jitted core), and `simulate_images_jax` batches it —
the batch is where a GPU wins big, since the per-sim `N_MAX` cost runs in
parallel. The coordinate transform is back in jax and matches the old numpy port
to float precision. The one modelling choice that differs from a naive
star-by-star pipeline is the selection cut, noted at its definition.

`N_MAX` must cover the star count across the prior or heavy streams get truncated
(their count channel saturates) — see the `N_MAX` note below; it is the direct
speed<->coverage knob and scales CPU cost ~linearly.

Falcon imports this module inside its Ray workers (see `paths.imports` in the
config), so everything here must be importable with no side effects beyond the
one-off constants below.
"""

import functools
import math
import secrets

import jax
import jax.numpy as jnp

# The physics package. Importing sstrax pulls in jax. `PRIOR_LIST` and
# `Parameters` come from sstrax.constants; the ODE / stream internals are what we
# vmap over instead of calling the Python-loop `simulate_stream`.
import sstrax
from sstrax.constants import PRIOR_LIST  # noqa: F401  (re-exported for convenience)
from sstrax.ode import dynamics_solver, mass_solver
from sstrax.stream import init_stripping, sample_trace


# =============================================================================
# Simulator numbers — lifted verbatim from albatross' example config
# (undark-lab/albatross:examples/configs/example_config.txt). Do not invent
# new ones; these are the ranges/errors the pipeline was designed around.
# =============================================================================

# The 16 sstrax parameters, in sstrax order:
#   xc yc zc vxc vyc vzc age msat xi0 alpha rh mbar sigv lrelease lmatch stripnear
# Prior ranges. Mass is handled in log10 (name "logmsat") the way albatross does
# it, because msat spans decades; everything else is sampled linearly.
#
# `age` is capped at 2000 Myr (albatross used 5000): a tighter age prior sheds
# far fewer stars, which lets `N_MAX` — and hence the simulation cost — drop (see
# the N_MAX note). The fiducial below is moved to sit cleanly inside this prior so
# the observation stays recoverable.
PRIOR_RANGES = {
    "xc": (10.0, 14.0),
    "yc": (0.1, 2.5),
    "zc": (6.0, 8.0),
    "vxc": (90.0, 115.0),
    "vyc": (-280.0, -230.0),
    "vzc": (-120.0, -80.0),
    "age": (500.0, 2000.0),
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
# is not being inferred. `age` sits cleanly inside its (500, 2000) prior so the
# default (age, logmsat) posterior can peak on the truth rather than rail against
# a boundary.
TRUE_VALUES = {
    "xc": 11.8, "yc": 0.79, "zc": 6.4,
    "vxc": 109.5, "vyc": -254.5, "vzc": -90.3,
    "age": 1500.0, "logmsat": 4.05,
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

# Fixed star-array capacity. The whole trick is that this is *constant*, so the
# vmapped simulator compiles once and never re-traces. It must cover the star
# count across the prior or heavy streams get truncated (their count channel
# saturates). Cost on CPU scales ~linearly with it, so track it to the prior.
#
# N_stars is dominated by disruption age and progenitor mass. For the default
# (age<=2000, logmsat<=4.5, mbar=3) task the worst-corner count is ~1390, so 1536
# is a safe default that is exact everywhere in this prior. For reference, if you
# widen the age prior back out:
#     age<=2000 -> ~1390   (N_MAX=1536, the default here)
#     age<=3000 -> ~2050   (N_MAX=2048)
#     age<=5000 -> ~3250   (N_MAX=4096)
# (Halving mbar's lower bound roughly triples these; on a GPU the batch runs in
# parallel so N_MAX is close to free.)
N_MAX = 1536

# Number of uniform field-star background points (static -> keeps shapes fixed).
_N_BG = int(math.floor(ERRORS["total_background"] * ERRORS["background_removal"]))

# The small, intuitive parameter block notebook 3 infers by default: the
# stream's disruption age and its progenitor mass. Both visibly reshape the
# stream (older / heavier -> longer, denser), so the 2-D posterior is easy to
# read against the truth. The hackathon is about growing this list.
DEFAULT_INFER = ["age", "logmsat"]


# =============================================================================
# 1. sstrax parameters  <->  inference vector, and jax seeding
# =============================================================================

def _key(key=None, seed=None):
    """Resolve a jax PRNGKey. Pass an explicit `key` or an int `seed`; with
    neither we draw fresh entropy (via `secrets`, so no numpy) exactly the way
    albatross decorrelated falcon's parallel workers."""
    if key is not None:
        return key
    if seed is None:
        seed = secrets.randbits(31)
    return jax.random.PRNGKey(int(seed))


def params_from_vector(z, infer_params):
    """Build an sstrax `Parameters` from an inference vector `z`.

    `z[i]` is the value of `infer_params[i]`; every other parameter is held at
    its fiducial `TRUE_VALUES`. The name "logmsat" maps onto `msat = 10**value`.
    """
    z = jnp.asarray(z, dtype=float).reshape(-1)
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
# 2. halo -> GD1, in jax
# =============================================================================
# The whole halo -> GD1-cartesian map is affine (a rotation + a translation), so
# its Jacobian is a single constant 3x3 matrix `_C` that we extract once (below).
# The velocity transform is that constant matrix followed by the analytic
# Jacobian of the (r, phi1, phi2) spherical map. Matches sstrax.projection to
# float precision.

def _halo_to_gd1cart(Xhalo):
    """(N, 3) halo-frame positions -> (N, 3) cartesian GD1 positions."""
    x, y, z = Xhalo[:, 0], Xhalo[:, 1], Xhalo[:, 2]
    xsun, ysun, zsun = 8.0 - x, y, z                       # halo_to_sun
    r = jnp.sqrt(xsun ** 2 + ysun ** 2 + zsun ** 2)        # sun_to_gal
    b = jnp.arcsin(zsun / r)
    l = jnp.arctan2(ysun, xsun)
    # gal_to_equat (Galactic -> equatorial rotation via NGP constants)
    dNGP = 27.12825118085622 * jnp.pi / 180.0
    lNGP = 122.9319185680026 * jnp.pi / 180.0
    aNGP = 192.85948 * jnp.pi / 180.0
    sb, cb = jnp.sin(b), jnp.cos(b)
    sl, cl = jnp.sin(lNGP - l), jnp.cos(lNGP - l)
    alpha = jnp.arctan((cb * sl) / (jnp.cos(dNGP) * sb - jnp.sin(dNGP) * cb * cl)) + aNGP
    delta = jnp.arcsin(jnp.sin(dNGP) * sb + jnp.cos(dNGP) * cb * cl)
    ca, sa, cd, sd = jnp.cos(alpha), jnp.sin(alpha), jnp.cos(delta), jnp.sin(delta)
    # equat_to_gd1cart (fixed rotation matrix)
    xg = r * (-0.4776303088 * ca * cd - 0.1738432154 * sa * cd + 0.8611897727 * sd)
    yg = r * (0.510844589 * ca * cd - 0.8524449229 * sa * cd + 0.111245042 * sd)
    zg = r * (0.7147776536 * ca * cd + 0.4930681392 * sa * cd + 0.4959603976 * sd)
    return jnp.stack([xg, yg, zg], axis=1)


# Constant Jacobian of the affine halo -> GD1-cartesian map (extracted once).
_ORIGIN = _halo_to_gd1cart(jnp.zeros((1, 3)))[0]
_C = jnp.stack(
    [_halo_to_gd1cart(jnp.eye(3)[j:j + 1])[0] - _ORIGIN for j in range(3)], axis=1
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
    stars = jnp.asarray(stars, dtype=float)
    Xh, Vh = stars[:, :3], stars[:, 3:]
    Xc = _halo_to_gd1cart(Xh)          # cartesian GD1 positions
    Vc = Vh @ _C.T                     # cartesian GD1 velocities (affine -> linear)
    x, y, z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
    r = jnp.sqrt(x ** 2 + y ** 2 + z ** 2)
    rho2 = x ** 2 + y ** 2
    rho = jnp.sqrt(rho2)
    phi1 = jnp.arctan2(y, x)
    phi2 = jnp.arcsin(z / r)
    vx, vy, vz = Vc[:, 0], Vc[:, 1], Vc[:, 2]
    # analytic Jacobian of (r, phi1, phi2) w.r.t. cartesian, applied to velocity
    dr = (x * vx + y * vy + z * vz) / r
    dphi1 = (-y * vx + x * vy) / rho2
    dphi2 = (-x * z * vx - y * z * vy + rho2 * vz) / (r ** 2 * rho)
    return jnp.stack([
        r,                                 # dist  [kpc]
        phi1 * 180.0 / jnp.pi,             # phi1  [deg]
        phi2 * 180.0 / jnp.pi,             # phi2  [deg]
        dr * _KPCMYR_TO_KMS,               # vrad  [km/s]
        dphi1 / r * _RADMYR_TO_MASYR,      # pm_phi1_cosphi2 [mas/yr]
        dphi2 / r * _RADMYR_TO_MASYR,      # pm_phi2 [mas/yr]
    ], axis=1)


# =============================================================================
# 3. noise, background, binning  (jax; weighted so shapes stay fixed)
# =============================================================================
# Column layout of a GD1 row and its (lo, hi) binning window, in one place so the
# noise vector, background bounds, and histograms all stay in sync.
_GD1_COLS = ("dist", "phi1", "phi2", "vrad", "pm_phi1_cosphi2", "pm_phi2")
_ERR_VEC = jnp.array([ERRORS[c] for c in _GD1_COLS])
_BIN_LO = jnp.array([BINNING[c][0] for c in _GD1_COLS])
_BIN_HI = jnp.array([BINNING[c][1] for c in _GD1_COLS])


def _add_noise(gd1, w, key):
    """Add per-observable Gaussian errors and apply the selection cut.

    Returns `(gd1, w)` with the errors added and `w` zeroed for de-selected
    stars. Selection keeps each star independently with prob `stream_selection`
    (a Binomial count), whereas a fixed-count draw would keep exactly
    floor(fraction * N); both mean "keep ~95%" and differ by ~sqrt(N p (1-p)) ~
    0.7%, far below the per-bin counting noise. Generate the observation and the
    training sims with this same model and it is a non-issue.
    """
    k_noise, k_sel = jax.random.split(key)
    gd1 = gd1 + jax.random.normal(k_noise, gd1.shape) * _ERR_VEC
    keep = jax.random.uniform(k_sel, (gd1.shape[0],)) < ERRORS["stream_selection"]
    return gd1, w * keep.astype(w.dtype)


def _sample_background(key, nbins):
    """Uniform Milky-Way field-star contamination binned into (3, nbins, nbins).

    `_N_BG` stars scattered uniformly across the binning ranges — an irreducible,
    parameter-independent floor under the stream."""
    bg = _BIN_LO + jax.random.uniform(key, (_N_BG, 6)) * (_BIN_HI - _BIN_LO)
    return _bin_stream(bg, jnp.ones(_N_BG), nbins)


def _whist2d(x, y, w, rng_x, rng_y, nbins):
    """Weighted 2-D count histogram, fixed shape. Matches np.histogram2d exactly
    (same right-open bin edges); `w` are per-star weights (0 for padding /
    de-selected stars)."""
    ix = jnp.floor((x - rng_x[0]) / (rng_x[1] - rng_x[0]) * nbins).astype(jnp.int32)
    iy = jnp.floor((y - rng_y[0]) / (rng_y[1] - rng_y[0]) * nbins).astype(jnp.int32)
    inb = (ix >= 0) & (ix < nbins) & (iy >= 0) & (iy < nbins)
    flat = jnp.where(inb, ix * nbins + iy, 0)
    ww = jnp.where(inb, w, 0.0)
    return jax.ops.segment_sum(ww, flat, num_segments=nbins * nbins).reshape(nbins, nbins)


def _bin_stream(gd1, w, nbins):
    """(N, 6) GD1 stars + per-star weights -> (3, nbins, nbins) count stack."""
    dist, phi1, phi2, vrad, pm1, pm2 = (gd1[:, i] for i in range(6))
    sky = _whist2d(phi1, phi2, w, BINNING["phi1"], BINNING["phi2"], nbins)
    pm = _whist2d(pm1, pm2, w, BINNING["pm_phi1_cosphi2"], BINNING["pm_phi2"], nbins)
    dv = _whist2d(dist, vrad, w, BINNING["dist"], BINNING["vrad"], nbins)
    return jnp.stack([sky, pm, dv])


def bin_stream(gd1, nbins=NBINS):
    """(N, 6) GD1 stars -> (3, nbins, nbins) stack of 2-D count histograms:
      channel 0: (phi1, phi2)                  — the stream on the sky
      channel 1: (pm_phi1_cosphi2, pm_phi2)    — proper motions
      channel 2: (dist, vrad)                  — distance vs radial velocity
    """
    gd1 = jnp.asarray(gd1, dtype=float)
    return _bin_stream(gd1, jnp.ones(gd1.shape[0]), nbins).astype(jnp.float32)


# =============================================================================
# 4. one full forward simulation (jitted, fixed-N)
# =============================================================================

@functools.partial(jax.jit, static_argnums=(2, 3, 4))
def simulate_image_jax(key, params, n_max=N_MAX, nbins=NBINS, with_background=True):
    """Full forward model in jax: `sstrax.Parameters` -> (3, nbins, nbins) image.

    Runs the stripped-star evolution (all `n_max` stars in one vmapped kernel,
    real stars carry weight 1 and padding weight 0), rotates to GD1, adds noise
    and the field-star background, and bins — every stage jax, so this whole
    function compiles once and runs on whatever device jax sees, GPU included.
    `key` seeds both the stream and the observational noise/background; `n_max`,
    `nbins`, `with_background` are static.

    Truncation: if the true star count exceeds `n_max` the extra stars are
    dropped (the count channel saturates at `n_max`); keep `n_max` above the
    prior's star count to avoid that — see the `N_MAX` note above.
    """
    k_star, k_noise, k_bg = jax.random.split(key, 3)

    # sstrax internals, exactly as its simulate_stream sets them up...
    clust_sol = dynamics_solver(params.cluster_final, params.age, 0.0,
                                dense=True, maxstep_warnings=False)
    mass_sol = mass_solver(params, clust_sol, maxstep_warnings=False)
    nstars_f = jnp.floor((params.msat - mass_sol.evaluate(params.age)) / params.mbar)
    strip = init_stripping(params, mass_sol)

    # ...but evolve all n_max stars in ONE vmapped kernel instead of a Python loop
    keys = jax.random.split(k_star, n_max)
    stars = jax.vmap(sample_trace, in_axes=(0, None, None, None, None))(
        keys, strip, clust_sol, mass_sol, params)                 # (n_max, 6)
    w = (jnp.arange(n_max) < nstars_f).astype(stars.dtype)        # real-star mask

    gd1 = stars_to_gd1(stars)
    gd1, w = _add_noise(gd1, w, k_noise)
    image = _bin_stream(gd1, w, nbins)
    if with_background:
        image = image + _sample_background(k_bg, nbins)
    return image.astype(jnp.float32)


def simulate_image(z, infer_params=DEFAULT_INFER, nbins=NBINS,
                   with_background=True, key=None, seed=None, n_max=N_MAX):
    """Full forward model: inference vector `z` -> (3, nbins, nbins) image.

    Convenience wrapper around `simulate_image_jax`: turns `z` into an sstrax
    `Parameters` (holding everything not in `infer_params` at its fiducial) and
    seeds a jax key. Pass an explicit `key`/`seed` for reproducibility, else
    fresh entropy is drawn. Returns a jax array.
    """
    params = params_from_vector(z, infer_params)
    return simulate_image_jax(_key(key, seed), params, n_max, nbins, with_background)


def _params_batch_from_Z(Z, infer_params):
    """Stack per-row `Parameters` into one batched (vmappable) Parameters pytree."""
    plist = [params_from_vector(z, infer_params) for z in Z]
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *plist)


def simulate_images_jax(Z, infer_params=DEFAULT_INFER, keys=None, seed=None,
                        n_max=N_MAX, nbins=NBINS, with_background=True):
    """Batched forward model: (B, d) inference vectors -> (B, 3, nbins, nbins).

    vmaps `simulate_image_jax` over the batch. On a GPU the B simulations run in
    parallel — this is the path to use when you want the simulator "as fast as
    possible" and have an accelerator. Pass `keys` (a (B,) jax PRNGKey array) or
    a `seed` for reproducibility. Returns a host array (via `jax.device_get`) so
    it drops straight into falcon.
    """
    Z = jnp.asarray(Z, dtype=float).reshape(-1, len(infer_params))
    if keys is None:
        keys = jax.random.split(_key(seed=seed), Z.shape[0])
    pbatch = _params_batch_from_Z(Z, infer_params)
    batched = jax.vmap(
        lambda k, p: simulate_image_jax(k, p, n_max, nbins, with_background),
        in_axes=(0, 0),
    )
    return jax.device_get(batched(keys, pbatch))


# =============================================================================
# 5. falcon simulator node
# =============================================================================

class StreamImage:
    """Falcon data node: parameters -> a `(3, nbins, nbins)` GD1 image.

    A falcon simulator node is any object with a
    `simulate_batch(self, batch_size, *parents)` method returning an array with a
    leading batch axis. The first parent, `z`, is the inference vector (the subset
    of parameters named by `infer_params`, in that order). An optional second
    parent, `z_nuis`, carries *nuisance* parameters named by `nuisance_params`:
    give that node a prior but no `evidence:` in the config, and falcon samples it
    and marginalises over it automatically (see notebook 2's nuisance example).
    Everything not in either list stays at its fiducial value.

    The batch is run through the batched jax model (`simulate_images_jax`); point
    jax at a GPU and the whole batch simulates in parallel. `n_max` is the fixed
    star capacity (see the `N_MAX` note).
    """

    def __init__(self, infer_params=DEFAULT_INFER, nuisance_params=None,
                 nbins=NBINS, with_background=True, n_max=N_MAX):
        self.infer_params = list(infer_params)
        self.nuisance_params = list(nuisance_params) if nuisance_params else []
        self.nbins = int(nbins)
        self.with_background = bool(with_background)
        self.n_max = int(n_max)

    def simulate_batch(self, batch_size, z, z_nuis=None):
        names = self.infer_params + self.nuisance_params
        z = jnp.asarray(z, dtype=float).reshape(-1, len(self.infer_params))
        if self.nuisance_params:
            z_nuis = jnp.asarray(z_nuis, dtype=float).reshape(-1, len(self.nuisance_params))
            Z = jnp.concatenate([z, z_nuis], axis=1)
        else:
            Z = z
        return simulate_images_jax(
            Z, infer_params=names, n_max=self.n_max, nbins=self.nbins,
            with_background=self.with_background,
        )


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
