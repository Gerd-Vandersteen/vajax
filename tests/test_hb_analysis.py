"""Tests for the Harmonic Balance spectral machinery (AFM grids + FFT-orthogonal transforms).

The legacy one-sided grid builders / least-squares APFT were replaced by the
artificial-frequency-mapping (AFM) path (VASAX Phase 9); the deeper property suite
(bin bijection sweeps, DDT exactness on every grid tone, single-FFT == matrices)
lives with the consumer in VASAX ``tests/test_hb_afm.py``.
"""

import jax.numpy as jnp
import pytest

from vajax.analysis.hb import (
    AFMGrid,
    HBConfig,
    HBResult,
    afm_fd_to_td,
    afm_td_to_fd,
    build_afm_grid,
    build_apft_matrices,
    complex_to_phasors,
    phasors_to_complex,
)


class TestAFMGrid:
    def test_single_tone(self):
        afm = build_afm_grid(HBConfig(freq=[1e3], nharm=4, truncation="box"))
        assert isinstance(afm, AFMGrid)
        assert afm.nf == 5 and afm.NT == 9 and afm.a_coeffs == (1,)
        assert jnp.allclose(afm.freqs_signed, jnp.arange(5) * 1e3)

    def test_two_tone_box(self):
        afm = build_afm_grid(HBConfig(freq=[1e3, 1.5e3], nharm=[2, 2], truncation="box"))
        assert afm.NT == 25 and afm.nf == 13
        # difference frequency present (the legacy one-sided grid lacked it)
        assert bool(jnp.any(jnp.isclose(jnp.abs(afm.freqs_signed), 500.0)))

    def test_two_tone_diamond(self):
        afm = build_afm_grid(HBConfig(freq=[1e3, 1.5e3], nharm=2, truncation="diamond"))
        assert afm.NT == 13 and afm.nf == 7
        assert list(afm.bins) == list(range(7))

    def test_dc_first(self):
        afm = build_afm_grid(HBConfig(freq=[1e3, 1.5e3], nharm=[3, 3], truncation="box"))
        assert float(afm.freqs_signed[0]) == 0.0 and int(afm.bins[0]) == 0

    def test_three_tones_raise(self):
        with pytest.raises(NotImplementedError):
            build_afm_grid(HBConfig(freq=[1e3, 2e3, 3e3], nharm=2, truncation="box"))

    def test_unknown_truncation(self):
        with pytest.raises(ValueError):
            build_afm_grid(HBConfig(freq=[1e3, 2e3], nharm=2, truncation="banana"))


class TestAPFTMatrices:
    def _afm(self):
        return build_afm_grid(HBConfig(freq=[1e3, 1.5e3], nharm=[3, 3], truncation="box"))

    def test_matrix_dimensions(self):
        afm = self._afm()
        APFT, IAPFT, DDT = build_apft_matrices(afm)
        assert APFT.shape == (2 * afm.nf - 1, afm.NT)
        assert IAPFT.shape == (afm.NT, 2 * afm.nf - 1)
        assert DDT.shape == (afm.NT, afm.NT)

    def test_apft_iapft_inverse(self):
        afm = self._afm()
        APFT, IAPFT, _ = build_apft_matrices(afm)
        err = jnp.max(jnp.abs(IAPFT @ APFT - jnp.eye(afm.NT)))
        assert float(err) < 1e-12

    def test_dc_preserved(self):
        afm = self._afm()
        APFT, _, _ = build_apft_matrices(afm)
        spec = APFT @ jnp.full(afm.NT, 2.5)
        assert jnp.isclose(spec[0], 2.5) and float(jnp.max(jnp.abs(spec[1:]))) < 1e-12

    def test_ddt_dc_zero(self):
        afm = self._afm()
        _, _, DDT = build_apft_matrices(afm)
        w_max = 2 * jnp.pi * jnp.max(jnp.abs(afm.freqs_signed))
        assert float(jnp.max(jnp.abs(DDT @ jnp.ones(afm.NT)))) < 1e-10 * float(w_max)

    def test_fft_transforms_match_matrices(self):
        afm = self._afm()
        APFT, IAPFT, _ = build_apft_matrices(afm)
        x = jnp.sin(jnp.linspace(0.3, 7.0, afm.NT))
        X = afm_td_to_fd(x, afm)
        assert float(jnp.max(jnp.abs(X - APFT @ x))) < 1e-13
        assert float(jnp.max(jnp.abs(afm_fd_to_td(X, afm) - IAPFT @ X))) < 1e-13


class TestPhasorConversion:
    def test_roundtrip(self):
        nf = 4
        phasors = jnp.array([1.0 + 0j, 0.5 - 0.25j, 0.1 + 0.3j, -0.2 + 0j])
        spec = complex_to_phasors(phasors)
        back = phasors_to_complex(spec, nf)
        assert jnp.allclose(back, phasors)

    def test_dc_real(self):
        spec = complex_to_phasors(jnp.array([1.5 + 0.7j, 0.2 + 0.1j]))
        assert jnp.isclose(spec[0], 1.5)


class TestHBConfig:
    def test_default_config(self):
        cfg = HBConfig()
        assert cfg.freq == [1e3] and cfg.nharm == 4 and cfg.truncation == "diamond"

    def test_custom_config(self):
        cfg = HBConfig(freq=[1e3, 1.5e3], nharm=[2, 8], truncation="box", sample_factor=2.0)
        assert cfg.freq == [1e3, 1.5e3] and cfg.nharm == [2, 8]


class TestHBResult:
    def test_result_fields(self):
        r = HBResult(frequencies=jnp.array([0.0, 1e3]), phasors={})
        assert not r.converged and r.iterations == 0
