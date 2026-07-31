"""Plotting helpers for the stellar-stream hackathon (notebook 3).

Matplotlib + numpy only — no `corner`, no seaborn — so nothing here can break a
fresh Colab or a virtual attendee's environment. Everything reads the binning /
prior numbers straight from `streams_model`, so plots always match the
simulator.
"""

import numpy as np
import matplotlib.pyplot as plt

import streams_model as sm

# Channel metadata: (title, x-label, y-label, x-range, y-range), matching the
# three histograms produced by streams_model.bin_stream.
_CHANNELS = [
    (r"sky: $\phi_1$ vs $\phi_2$", r"$\phi_1$ [deg]", r"$\phi_2$ [deg]",
     sm.BINNING["phi1"], sm.BINNING["phi2"]),
    (r"proper motion", r"$\mu_{\phi_1}\cos\phi_2$ [mas/yr]", r"$\mu_{\phi_2}$ [mas/yr]",
     sm.BINNING["pm_phi1_cosphi2"], sm.BINNING["pm_phi2"]),
    (r"distance vs radial velocity", r"dist [kpc]", r"$v_{\rm rad}$ [km/s]",
     sm.BINNING["dist"], sm.BINNING["vrad"]),
]


def plot_stream_orbit_and_sky(stars, gd1=None, title=None, ax=None):
    """Show one simulated stream two ways.

    Left/middle: the raw stars in the Milky-Way (halo) frame, x-y and x-z
    projections in kpc — the physical stream wrapping around the Galaxy.
    Right: the same stars projected onto the GD1 sky coordinates
    (phi1, phi2) — what an observer actually measures.

    `stars` is the (N, 6) halo-frame phase space from
    `sstrax.simulate_stream`; `gd1` (optional) the (N, 6) GD1 observables, else
    it is computed here.
    """
    stars = np.asarray(stars)
    if gd1 is None:
        gd1 = sm.stars_to_gd1(stars)
    if ax is None:
        fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    x, y, z = stars[:, 0], stars[:, 1], stars[:, 2]
    ax[0].scatter(x, y, s=2, alpha=0.3, color="C0")
    ax[0].scatter([0], [0], marker="*", s=180, color="k", label="Galactic centre")
    ax[0].set_xlabel("x [kpc]"); ax[0].set_ylabel("y [kpc]")
    ax[0].set_title("halo frame (x–y)"); ax[0].legend(fontsize=8)
    ax[0].set_aspect("equal", adjustable="datalim")

    ax[1].scatter(x, z, s=2, alpha=0.3, color="C0")
    ax[1].scatter([0], [0], marker="*", s=180, color="k")
    ax[1].set_xlabel("x [kpc]"); ax[1].set_ylabel("z [kpc]")
    ax[1].set_title("halo frame (x–z)")
    ax[1].set_aspect("equal", adjustable="datalim")

    ax[2].scatter(gd1[:, 1], gd1[:, 2], s=2, alpha=0.3, color="C3")
    ax[2].set_xlabel(r"$\phi_1$ [deg]"); ax[2].set_ylabel(r"$\phi_2$ [deg]")
    ax[2].set_xlim(*sm.BINNING["phi1"]); ax[2].set_ylim(*sm.BINNING["phi2"])
    ax[2].set_title("GD1 sky coordinates")

    if title:
        ax[0].figure.suptitle(title, y=1.03)
    ax[0].figure.tight_layout()
    return ax


def plot_channels(image, title=None, cmap="magma", axes=None):
    """Show a `(3, nbins, nbins)` stream image as three labelled panels.

    This is exactly what the CNN embedding sees — three 2-D count histograms.
    """
    image = np.asarray(image)
    assert image.shape[0] == 3, f"expected (3, nbins, nbins), got {image.shape}"
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for ch, ax in enumerate(axes):
        ttl, xl, yl, xr, yr = _CHANNELS[ch]
        ax.imshow(image[ch].T, origin="lower", aspect="auto", cmap=cmap,
                  extent=[xr[0], xr[1], yr[0], yr[1]])
        ax.set_title(ttl); ax.set_xlabel(xl); ax.set_ylabel(yl)
    if title:
        axes[0].figure.suptitle(title, y=1.04)
    axes[0].figure.tight_layout()
    return axes


def plot_posterior(samples, infer_params, truth=None, prior_ranges=None,
                   color="C0", title=None):
    """Corner-style view of a falcon posterior over `infer_params`.

    `samples` is `(N, D)` with `D == len(infer_params)`. Handles D == 1 (a
    histogram), D == 2 (a scatter with 1-D marginals), and D > 2 (a lower-
    triangular pairs grid). Truth values and the prior box are overlaid when
    given.
    """
    samples = np.atleast_2d(np.asarray(samples))
    if samples.shape[1] != len(infer_params):
        samples = samples.T
    D = len(infer_params)
    ranges = prior_ranges or sm.PRIOR_RANGES
    labels = list(infer_params)

    if D == 1:
        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        ax.hist(samples[:, 0], bins=40, density=True, color=color, alpha=0.7)
        if truth is not None:
            ax.axvline(truth[0], color="C3", lw=2, label="truth"); ax.legend(fontsize=8)
        ax.set_xlabel(labels[0]); ax.set_xlim(*ranges[labels[0]])
        if title:
            ax.set_title(title)
        fig.tight_layout()
        return fig

    fig, axes = plt.subplots(D, D, figsize=(2.6 * D, 2.6 * D))
    axes = np.atleast_2d(axes)
    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j > i:
                ax.axis("off"); continue
            if i == j:
                ax.hist(samples[:, i], bins=35, density=True, color=color, alpha=0.7)
                if truth is not None:
                    ax.axvline(truth[i], color="C3", lw=1.5)
                ax.set_xlim(*ranges[labels[i]])
                ax.set_yticks([])
            else:
                ax.scatter(samples[:, j], samples[:, i], s=3, alpha=0.15, color=color)
                if truth is not None:
                    ax.plot(truth[j], truth[i], "*", color="C3", ms=12)
                ax.set_xlim(*ranges[labels[j]]); ax.set_ylim(*ranges[labels[i]])
            if i == D - 1:
                ax.set_xlabel(labels[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(labels[i])
            else:
                if not (i == j):
                    ax.set_yticklabels([])
    if title:
        fig.suptitle(title, y=1.0)
    fig.tight_layout()
    return fig
