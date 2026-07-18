"""Harmonic Balance spectral machinery for VAJAX — AFM grids + FFT-orthogonal transforms.

This module ships the **spectral building blocks** consumed by a downstream HB solver
(VASAX ``HBSolver``): the artificial-frequency-mapped (AFM) harmonic grid, the exact
APFT/IAPFT/DDT matrices on it, and the single-FFT transform pair.  There is no solver
here — residual assembly and the Newton solve live with the consumer.

Artificial frequency mapping (Kundert): an injective linear map
``lambda(k) = sum_j a_j*k_j (mod NT)`` sends the two-sided 1-/2-tone truncation (box or
diamond) onto the harmonics of one artificial fundamental sampled at ``NT`` uniform
points.  FD<->TD is then a single length-``NT`` real FFT, exactly orthogonal for any
tone separation — the legacy least-squares APFT this replaced went singular for widely
separated tones and its one-sided grids lacked the intermod (difference) frequencies.

Physical interpretation:
- Circuit equation: f_resist(v) + dQ/dt = 0
- Time-domain collocation formulation: f_resist + DDT*Q = 0
- ``DDT = IAPFT @ Omega @ APFT`` with the SIGNED true frequencies in Omega.

References:
- K. S. Kundert, et al., "Steady-State Methods for Simulating Analog and
  Microwave Circuits", Kluwer Academic Publishers, 1990
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
from jax import Array

from vajax import get_float_dtype

logger = logging.getLogger(__name__)

# Constants
TWO_PI = 2.0 * jnp.pi


@dataclass
class HBConfig:
    """Harmonic Balance analysis configuration.

    Attributes:
        freq: Fundamental frequencies in Hz (list of floats)
        nharm: Number of harmonics for each fundamental (int or list)
        truncation: 'box' or 'diamond' truncation scheme
        sample_factor: Oversampling factor for the nonlinearity evaluation
            (1.0 = critical sampling, aliases out-of-band products; > 1 evaluates on a
            finer artificial grid and truncates them instead — build_afm_oversampled)
        max_iterations: Maximum NR iterations
        abstol: Absolute tolerance for convergence
        reltol: Relative tolerance for convergence
    """

    freq: List[float] = field(default_factory=lambda: [1e3])
    nharm: Union[int, List[int]] = 4
    truncation: str = "diamond"
    sample_factor: float = 1.0
    max_iterations: int = 100
    abstol: float = 1e-9
    reltol: float = 1e-6


@dataclass
class HBResult:
    """Harmonic Balance analysis results.

    Attributes:
        frequencies: Array of frequencies in the spectrum (Hz)
        phasors: Complex phasors for each node, shape (n_nodes, n_freqs)
        dc_voltages: DC operating point voltages
        converged: Whether the analysis converged
        iterations: Number of NR iterations
        max_residual: Maximum residual at convergence
    """

    frequencies: Array
    phasors: Dict[str, Array]  # node_name -> complex phasor array
    dc_voltages: Optional[Array] = None
    converged: bool = False
    iterations: int = 0
    max_residual: float = 0.0


# =============================================================================
# Artificial frequency mapping (AFM) — Kundert-style multi-tone HB grids
# (VASAX Phase 9 A1; replaces the legacy grid/collocation/LS-APFT path in A2)
# =============================================================================


@dataclass
class AFMGrid:
    """Artificial-frequency-mapped harmonic grid (Kundert AFM).

    An injective linear map ``lambda(k) = sum_j a_j*k_j (mod NT)`` sends the
    two-sided harmonic truncation onto the harmonics of one *artificial*
    fundamental sampled at ``NT`` uniform points, so FD<->TD is a single
    length-``NT`` real FFT and the transform is exactly orthogonal (no
    least-squares conditioning issues for widely separated tones).

    Canonicalization: one representative per conjugate pair ``{k, -k}``,
    chosen so its mapped bin lies in the rfft half-spectrum
    ``0..(NT-1)/2``.  (For box truncation this coincides with the
    "first non-zero index positive" rule; for diamond it need not — e.g.
    K=2 maps (1,-1) to bin 7 of 13, so its conjugate (-1,1) at bin 6 is
    canonical.)  The *true* frequency of a canonical point,
    ``sum_j k_j*f_j``, may therefore be negative; it is kept signed here
    (the cos/sin basis and the Omega operator are consistent with signed
    omega) and readouts expose ``|f|`` with the phasor conjugated for
    ``f < 0``.

    Attributes:
        fund_freqs: Fundamental frequencies in Hz (n_funds,)
        grid: (nf, n_funds) int array — canonical harmonic indices ``k``
        freqs_signed: (nf,) float array — true frequency ``sum_j k_j*f_j``
            of each canonical point (signed; DC first)
        bins: (nf,) int array — rfft bin of each canonical point.  The
            canonical set maps *bijectively* onto ``0..(NT-1)/2``, so after
            sorting this is exactly ``arange(nf)``.
        NT: Number of artificial time samples (odd; ``NT == 2*nf - 1``)
        a_coeffs: The map coefficients ``a_j`` (n_funds,)
        tone_phases: (n_funds, NT) float array — per-tone excitation phase
            vectors ``theta_j[n] = 2*pi*a_j*n/NT`` (no physical time exists
            in multi-tone AFM; single tone recovers ``t_n = theta_1[n] /
            (2*pi*f0)``)
        nf: Number of canonical frequencies (incl. DC)
    """

    fund_freqs: Tuple[float, ...]
    grid: Array
    freqs_signed: Array
    bins: Array
    NT: int
    a_coeffs: Tuple[int, ...]
    tone_phases: Array
    nf: int


def build_afm_grid(config: HBConfig) -> AFMGrid:
    """Build the AFM grid for a 1- or 2-tone HB configuration.

    Truncations (two-sided, canonical half stored):
    - box, ``nharm=[K1, K2]``: ``|k1| <= K1, |k2| <= K2``;
      map ``a = (2*K2+1, 1)``, ``NT = (2*K1+1)*(2*K2+1)`` (a flattened
      2-D DFT).  ``nf = (K2+1) + K1*(2*K2+1)``.
    - diamond, ``nharm=K``: ``|k1| + |k2| <= K``;
      ``NT = K^2 + (K+1)^2`` (the diamond tiles Z^2 under the lattice
      generated by (K+1, K) and (-K, K+1), |det| = NT); map ``a2 = 1``,
      ``a1 = (-K * modinv(K+1, NT)) mod NT``.  ``nf = K*(K+1) + 1``.
    - single tone (either truncation): the degenerate exact DFT,
      ``a = (1,)``, ``NT = 2*K + 1``.

    Raises:
        NotImplementedError: for more than 2 fundamentals.
        ValueError: for an unknown truncation scheme.
    """
    fund = tuple(float(f) for f in config.freq)
    n_funds = len(fund)
    if n_funds > 2:
        raise NotImplementedError("AFM supports 1 or 2 fundamentals (V1)")

    nharm = config.nharm
    if n_funds == 1:
        K = int(nharm if isinstance(nharm, int) else nharm[0])
        a = (1,)
        NT = 2 * K + 1
        lattice = [(k,) for k in range(-K, K + 1)]
    elif config.truncation == "box":
        K1, K2 = (
            (int(nharm), int(nharm))
            if isinstance(nharm, int)
            else (int(nharm[0]), int(nharm[1]))
        )
        a = (2 * K2 + 1, 1)
        NT = (2 * K1 + 1) * (2 * K2 + 1)
        lattice = [
            (k1, k2)
            for k1 in range(-K1, K1 + 1)
            for k2 in range(-K2, K2 + 1)
        ]
    elif config.truncation == "diamond":
        K = int(nharm if isinstance(nharm, int) else max(nharm))
        NT = K * K + (K + 1) * (K + 1)
        a1 = (-K * pow(K + 1, -1, NT)) % NT
        a = (a1, 1)
        lattice = [
            (k1, k2)
            for k1 in range(-K, K + 1)
            for k2 in range(-(K - abs(k1)), K - abs(k1) + 1)
        ]
    else:
        raise ValueError(f"Unknown truncation scheme: {config.truncation}")

    # Canonical half: DC + the representative of each conjugate pair whose
    # mapped bin lies in the rfft half-spectrum 0..(NT-1)/2.
    half = (NT - 1) // 2
    entries = []
    for k in lattice:
        b = sum(aj * kj for aj, kj in zip(a, k)) % NT
        if b == 0:
            if any(k):
                raise AssertionError(
                    f"AFM map degenerate: non-DC point {k} maps to bin 0"
                )
            continue  # DC handled explicitly below
        if b <= half:
            entries.append((b, k))
    entries.sort()

    bins_list = [0] + [b for b, _ in entries]
    grid_list = [(0,) * n_funds] + [k for _, k in entries]
    nf = len(grid_list)

    # Bijection guarantee: the two-sided truncation maps 1:1 onto Z_NT, so
    # the canonical half must cover the rfft bins exactly.  Failure here is
    # a builder bug, never a data condition.
    if bins_list != list(range(nf)) or NT != 2 * nf - 1:
        raise AssertionError(
            f"AFM bins not bijective onto the rfft half-spectrum: "
            f"nf={nf}, NT={NT}, bins={bins_list[:8]}..."
        )

    freqs_signed_list = [
        sum(kj * fj for kj, fj in zip(k, fund)) for k in grid_list
    ]

    # Physical |f| collisions (near-commensurate tones) merge readout labels
    # but do not affect the solve — the AFM bins stay distinct.  Warn only.
    abs_sorted = sorted(abs(f) for f in freqs_signed_list)
    fmax = abs_sorted[-1] if abs_sorted[-1] > 0 else 1.0
    for lo, hi in zip(abs_sorted, abs_sorted[1:]):
        if hi - lo <= 1e-9 * fmax:
            logger.warning(
                "AFM: near-equal |f| grid points (%.6g vs %.6g Hz) — "
                "readout labels merge; the solve is unaffected.",
                lo,
                hi,
            )
            break

    n = jnp.arange(NT, dtype=get_float_dtype())
    tone_phases = (
        TWO_PI * jnp.array(a, dtype=get_float_dtype())[:, None] * n[None, :] / NT
    )

    return AFMGrid(
        fund_freqs=fund,
        grid=jnp.array(grid_list, dtype=jnp.int32),
        freqs_signed=jnp.array(freqs_signed_list, dtype=get_float_dtype()),
        bins=jnp.array(bins_list, dtype=jnp.int32),
        NT=NT,
        a_coeffs=a,
        tone_phases=tone_phases,
        nf=nf,
    )


def _afm_basis(n_samples: int, nf: int) -> Tuple[Array, Array]:
    """(analysis, synthesis) of the ``nf``-bin artificial-harmonic basis on a uniform
    grid of ``n_samples`` points of the artificial period.

    After canonical bin-sorting the basis depends only on the grid length (bin k is the
    k-th artificial harmonic). Requires ``nf - 1 < n_samples / 2`` (no aliasing) — always
    true for the critical grid (``n_samples = NT = 2nf−1``) and any finer one.
    Spectrum layout ``[DC, Re1, Im1, ...]``; analysis = exact DFT rows (DC ``1/n``,
    cos/sin ``2/n`` — orthogonal, no inversion).
    """
    dtype = get_float_dtype()
    n = jnp.arange(n_samples, dtype=dtype)
    k = jnp.arange(1, nf, dtype=dtype)
    theta = TWO_PI * jnp.outer(n, k) / n_samples  # (n_samples, nf-1)
    C = jnp.cos(theta)
    S = jnp.sin(theta)

    cs = jnp.stack([C, S], axis=2).reshape(n_samples, 2 * (nf - 1))
    synthesis = jnp.concatenate([jnp.ones((n_samples, 1), dtype=dtype), cs], axis=1)

    cs_t = jnp.stack([C.T, S.T], axis=1).reshape(2 * (nf - 1), n_samples)
    analysis = jnp.concatenate(
        [jnp.full((1, n_samples), 1.0 / n_samples, dtype=dtype),
         (2.0 / n_samples) * cs_t], axis=0
    )
    return analysis, synthesis


def _afm_omega(afm: AFMGrid) -> Array:
    """The real j·ω block matrix: per canonical frequency the 2x2 rotation with SIGNED
    true omega (``[Re, Im] -> omega * [-Im, Re]``)."""
    nf = afm.nf
    Omega = jnp.zeros((2 * nf - 1, 2 * nf - 1), dtype=get_float_dtype())
    for j in range(1, nf):
        omega_j = TWO_PI * afm.freqs_signed[j]
        Omega = Omega.at[2 * j - 1, 2 * j].set(-omega_j)
        Omega = Omega.at[2 * j, 2 * j - 1].set(omega_j)
    return Omega


def _build_afm_apft_matrices(afm: AFMGrid) -> Tuple[Array, Array, Array]:
    """APFT/IAPFT/DDT on the critical AFM grid — the exact (orthogonal) mapped DFT."""
    APFT, IAPFT = _afm_basis(afm.NT, afm.nf)
    DDT = IAPFT @ _afm_omega(afm) @ APFT
    return APFT, IAPFT, DDT


def build_afm_oversampled(afm: AFMGrid, sample_factor: float
                          ) -> Tuple[Array, Array, Array, int]:
    """Oversampled-evaluation operators ``(U, P, D_os, NT_os)`` for a square HB residual.

    The unknowns stay the ``NT`` critical samples; the nonlinearity is evaluated on a
    finer uniform artificial grid of ``NT_os = next odd ≥ ceil(sample_factor·NT)`` points
    and projected back, so out-of-band products are **truncated instead of aliased**:

    - ``U  = IAPFT_os @ APFT_c``  (NT_os, NT): band-limited interpolation of the critical
      samples (and of the tone excitation) onto the oversampled grid;
    - ``P  = IAPFT_c @ APFT_os``  (NT, NT_os): de-aliasing projection of the evaluated
      nonlinearity back onto the critical samples;
    - ``D_os = IAPFT_c @ Omega @ APFT_os``: the projecting spectral derivative.

    ``F = P @ i(U·Y) + D_os @ q(U·Y)`` is still square in ``Y``; ``sample_factor <= 1``
    degenerates to ``U = P = I``, ``D_os = DDT`` (asserted equal by the caller's tests).
    """
    NT, nf = afm.NT, afm.nf
    NT_os = max(NT, int(-(-sample_factor * NT // 1)))  # ceil
    if NT_os % 2 == 0:
        NT_os += 1
    APFT_c, IAPFT_c = _afm_basis(NT, nf)
    APFT_os, IAPFT_os = _afm_basis(NT_os, nf)
    U = IAPFT_os @ APFT_c
    P = IAPFT_c @ APFT_os
    D_os = IAPFT_c @ _afm_omega(afm) @ APFT_os
    return U, P, D_os, NT_os


def afm_td_to_fd(x: Array, afm: AFMGrid) -> Array:
    """Time samples -> APFT spectrum via a single real FFT.

    ``x`` has the time axis first: shape ``(NT,)`` or ``(NT, ...)``.
    Returns ``(2*nf-1, ...)`` in the ``[DC, Re1, Im1, ...]`` layout,
    identical to ``APFT @ x``.
    """
    NT = afm.NT
    X = jnp.fft.rfft(x, axis=0)
    dc = X[0].real[None] / NT
    re = 2.0 * X[1:].real / NT
    im = -2.0 * X[1:].imag / NT
    cs = jnp.stack([re, im], axis=1).reshape((2 * (afm.nf - 1),) + x.shape[1:])
    return jnp.concatenate([dc, cs], axis=0).astype(get_float_dtype())


def afm_fd_to_td(spectrum: Array, afm: AFMGrid) -> Array:
    """APFT spectrum -> time samples via a single inverse real FFT.

    ``spectrum`` has the ``[DC, Re1, Im1, ...]`` layout on the first axis:
    shape ``(2*nf-1,)`` or ``(2*nf-1, ...)``.  Returns ``(NT, ...)``,
    identical to ``IAPFT @ spectrum``.
    """
    NT = afm.NT
    Z0 = (spectrum[0] * NT).astype(jnp.complex128)[None]
    Zk = (spectrum[1::2] - 1j * spectrum[2::2]) * (NT / 2.0)
    Z = jnp.concatenate([Z0, Zk], axis=0)
    return jnp.fft.irfft(Z, n=NT, axis=0).astype(get_float_dtype())


def build_apft_matrices(
    frequencies: Union[Array, AFMGrid],
    timepoints: Optional[Array] = None,
) -> Tuple[Array, Array, Array]:
    """Build Almost Periodic Fourier Transform matrices.

    APFT transforms time-domain values to frequency-domain phasors.
    IAPFT transforms frequency-domain phasors to time-domain values.
    DDT is the time derivative operator in time domain.

    Preferred form: pass an :class:`AFMGrid` (single argument) — the exact
    FFT-orthogonal mapped-DFT construction.  The legacy
    ``(frequencies, timepoints)`` least-squares form is kept only until the
    Phase 9 A2 switch and is removed then.

    Args:
        frequencies: An ``AFMGrid``, or (legacy) array of frequencies (Hz)
        timepoints: (legacy form only) array of collocation timepoints (s)

    Returns:
        (APFT, IAPFT, DDT) matrices where:
        - APFT: (2*nf-1, nt) real matrix, transforms TD to APFT spectrum
        - IAPFT: (nt, 2*nf-1) real matrix, transforms APFT spectrum to TD
        - DDT: (nt, nt) real matrix, time derivative operator
    """
    if isinstance(frequencies, AFMGrid):
        return _build_afm_apft_matrices(frequencies)
    nf = len(frequencies)
    nt = len(timepoints)

    # Build APFT matrix
    # For each frequency f_k, we have basis functions:
    # - DC (f=0): constant 1
    # - Non-DC: cos(2*pi*f*t), sin(2*pi*f*t)
    # The APFT spectrum has 2*nf-1 components:
    # [DC, Re(f1), Im(f1), Re(f2), Im(f2), ...]

    # Build basis matrix B: rows are timepoints, cols are basis functions
    B = jnp.zeros((nt, 2 * nf - 1), dtype=get_float_dtype())

    # DC component
    B = B.at[:, 0].set(1.0)

    # Non-DC components
    for k in range(1, nf):
        omega_k = 2.0 * jnp.pi * frequencies[k]
        # Cosine (real part)
        B = B.at[:, 2 * k - 1].set(jnp.cos(omega_k * timepoints))
        # Sine (imaginary part)
        B = B.at[:, 2 * k].set(jnp.sin(omega_k * timepoints))

    # APFT: least-squares projection from TD to FD
    # X_fd = (B^T B)^{-1} B^T x_td
    BtB = B.T @ B
    BtB_inv = jnp.linalg.inv(BtB)
    APFT = BtB_inv @ B.T

    # IAPFT: synthesis from FD to TD
    # x_td = B X_fd
    IAPFT = B

    # DDT operator: d/dt in time domain
    # In frequency domain: d/dt -> j*omega
    # DDT = IAPFT * diag(j*omega) * APFT
    # Since we work with real representation, this becomes:
    # For each frequency k: [Re, Im] -> omega * [-Im, Re]
    Omega = jnp.zeros((2 * nf - 1, 2 * nf - 1), dtype=get_float_dtype())
    for k in range(1, nf):
        omega_k = 2.0 * jnp.pi * frequencies[k]
        # [Re, Im] -> omega * [-Im, Re]
        Omega = Omega.at[2 * k - 1, 2 * k].set(-omega_k)  # Re <- -omega*Im
        Omega = Omega.at[2 * k, 2 * k - 1].set(omega_k)  # Im <- omega*Re

    DDT = IAPFT @ Omega @ APFT

    return APFT, IAPFT, DDT


def phasors_to_complex(apft_spectrum: Array, nf: int) -> Array:
    """Convert APFT real spectrum to complex phasors.

    Args:
        apft_spectrum: Real spectrum [DC, Re1, Im1, Re2, Im2, ...]
        nf: Number of frequencies

    Returns:
        Complex phasor array of shape (nf,)
    """
    phasors = jnp.zeros(nf, dtype=jnp.complex128)
    # DC
    phasors = phasors.at[0].set(apft_spectrum[0] + 0j)
    # Non-DC
    for k in range(1, nf):
        re = apft_spectrum[2 * k - 1]
        im = apft_spectrum[2 * k]
        phasors = phasors.at[k].set(re + 1j * im)
    return phasors


def complex_to_phasors(phasors: Array) -> Array:
    """Convert complex phasors to APFT real spectrum.

    Args:
        phasors: Complex phasor array of shape (nf,)

    Returns:
        Real spectrum [DC, Re1, Im1, Re2, Im2, ...]
    """
    nf = len(phasors)
    spectrum = jnp.zeros(2 * nf - 1, dtype=get_float_dtype())
    # DC
    spectrum = spectrum.at[0].set(jnp.real(phasors[0]))
    # Non-DC
    for k in range(1, nf):
        spectrum = spectrum.at[2 * k - 1].set(jnp.real(phasors[k]))
        spectrum = spectrum.at[2 * k].set(jnp.imag(phasors[k]))
    return spectrum

