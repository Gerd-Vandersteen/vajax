"""Collapse-decision guard extraction (openvaf_py) and its Python consumers.

Locks the collapse_guards:2 semantics:

- ``collapse_decision_outputs`` entries are ``(pair_idx, [(value_name, negate), ...])``
  where ``pair_idx`` is the TRUE ``CollapsePair`` index (resolved through
  ``node_collapse.hint()``, never a callback counter) and the conjunct list is the FULL
  path condition to the ``CollapseHint`` call site (a compound ``if / else-if / else``
  chain contributes every level, not just the innermost branch).
- Extra pairs (branch-current unknowns) mirror their source pair's guards; implicit-
  equation pairs carry their init-output bool as a single non-negated conjunct.
- ``_pairs_from_collapse_decisions`` OR-reduces multiple entries per pair and fires
  unguarded pairs unconditionally.

The Angelov-shaped compound guard here is the exact pattern the pre-fix extraction got
wrong (VASAX F1: ``di``<->``d`` collapsed even with Rd=3.3/Ld=300p, silently dropping the
drain resistor and inductor).
"""

import os

os.environ["JAX_ENABLE_X64"] = "true"

from pathlib import Path

import jax.numpy as jnp
import openvaf_py
import pytest

import openvaf_jax
from openvaf_jax.codegen.function_builder import InitFunctionBuilder
from vajax.analysis.openvaf_models import _pairs_from_collapse_decisions

PROJECT_ROOT = Path(__file__).parent.parent
ASMHEMT_VA = PROJECT_ROOT / "vajax" / "devices" / "models" / "asmhemt" / "asmhemt.va"

GUARDED_VA = """
`include "disciplines.vams"

module cguard(d, g, s);
    inout d, g, s;
    electrical d, g, s, di, si;

    parameter real Rd  = 0.0;
    parameter real Rd2 = 0.0;
    parameter real Ld  = 0.0;
    parameter real Rs  = 0.0;

    analog begin
        // keep the internal nodes live
        I(di, si) <+ 1e-3 * V(g, si);
        I(g, si)  <+ 1e-12 * V(g, si);

        // compound guard (the Angelov drain shape): collapse only when
        // !(Rd>0 || Rd2>0) && !(Ld>0)
        if ((Rd > 0.0) || (Rd2 > 0.0))
            V(di, d) <+ I(di, d) * (Rd + Rd2);
        else if (Ld > 0.0)
            V(di, d) <+ ddt(Ld * I(di, d));
        else
            V(di, d) <+ 0.0;

        // simple guard (the Angelov source shape): collapse only when !(Rs>0)
        if (Rs > 0.0)
            V(si, s) <+ I(si, s) * Rs;
        else
            V(si, s) <+ 0.0;
    end
endmodule
"""


@pytest.fixture(scope="module")
def cguard(tmp_path_factory):
    va = tmp_path_factory.mktemp("cguard") / "cguard.va"
    va.write_text(GUARDED_VA)
    modules = openvaf_py.compile_va(str(va))
    assert modules, "cguard.va failed to compile"
    return modules[0]


def _pair_by_names(module, n1, n2):
    """Return (pair_idx, (node1_idx, node2_idx)) for the pair with the given VA names."""
    for pd in module.get_dae_system()["collapsible_pairs"]:
        if {pd["node1_name"], pd["node2_name"]} == {n1, n2}:
            return pd["pair_idx"], (pd["node1_idx"], pd["node2_idx"])
    raise AssertionError(f"no collapsible pair ({n1},{n2})")


def _guards_of(module, pair_idx):
    return [conjs for idx, conjs in module.collapse_decision_outputs if idx == pair_idx]


class TestGuardStructure:
    def test_all_entries_well_formed(self, cguard):
        n = len(cguard.collapsible_pairs)
        assert n == cguard.num_collapsible
        for pair_idx, conjuncts in cguard.collapse_decision_outputs:
            assert 0 <= pair_idx < n
            for name, negate in conjuncts:
                assert isinstance(name, str) and name.startswith("v")
                assert isinstance(negate, bool)

    def test_simple_guard_single_negated_conjunct(self, cguard):
        pair_idx, _ = _pair_by_names(cguard, "si", "s")
        guards = _guards_of(cguard, pair_idx)
        assert len(guards) == 1
        assert len(guards[0]) == 1, f"simple if/else guard must be 1 conjunct: {guards}"
        assert guards[0][0][1] is True  # collapse on the FALSE side of (Rs > 0)

    def test_compound_guard_full_conjunction(self, cguard):
        """The pre-fix bug: only the innermost !(Ld>0) survived for the drain chain."""
        pair_idx, _ = _pair_by_names(cguard, "di", "d")
        guards = _guards_of(cguard, pair_idx)
        assert len(guards) == 1
        conjuncts = guards[0]
        assert len(conjuncts) >= 2, f"compound chain lost its outer conjuncts: {conjuncts}"
        assert all(neg is True for _, neg in conjuncts)

    def test_extra_pairs_mirror_source_guards(self, cguard):
        """flow(di,d)/flow(si,s) -> ground collapse exactly with their source pair."""
        hint_idx = {}
        for names in (("di", "d"), ("si", "s")):
            idx, _ = _pair_by_names(cguard, *names)
            hint_idx[names] = idx
        extra_idxs = [
            i
            for i, (_, n2) in enumerate(cguard.collapsible_pairs)
            if n2 == 4294967295 and i not in hint_idx.values()
        ]
        assert extra_idxs, "expected extra branch-current pairs"
        mirrored = 0
        for i in extra_idxs:
            guards = _guards_of(cguard, i)
            assert guards, f"extra pair {i} is unguarded (pre-fix behavior)"
            for names, hidx in hint_idx.items():
                if guards == _guards_of(cguard, hidx):
                    mirrored += 1
                    break
        assert mirrored == len(extra_idxs), "every extra pair must mirror a source guard"


class TestFiredPairs:
    """Behavioral: run init_fn and map decisions to fired pairs, both sides of the guard."""

    @pytest.fixture(scope="class")
    def init(self, cguard):
        translator = openvaf_jax.OpenVAFToJAX(cguard)
        init_fn, init_meta = translator.translate_init_array()
        defaults = init_meta.get("param_defaults", {}) or {}

        def fired(**overrides):
            over = {k.lower(): v for k, v in overrides.items()}
            vals = []
            for name in init_meta["param_names"]:
                key = name.lower()
                vals.append(float(over.get(key, defaults.get(name, 0.0) or 0.0)))
            _, decisions = init_fn(jnp.array(vals))
            return _pairs_from_collapse_decisions(
                decisions, list(cguard.collapsible_pairs), list(cguard.collapse_decision_outputs)
            )

        return fired

    def test_real_drain_branch_does_not_collapse(self, cguard, init):
        _, di_d = _pair_by_names(cguard, "di", "d")
        _, si_s = _pair_by_names(cguard, "si", "s")
        fired = init(Rd=3.3, Ld=300e-12, Rs=0.0)
        assert tuple(di_d) not in {tuple(p) for p in fired}, "F1: (di,d) collapsed with Rd>0"
        assert tuple(si_s) in {tuple(p) for p in fired}

    def test_inductor_only_drain_branch_does_not_collapse(self, cguard, init):
        _, di_d = _pair_by_names(cguard, "di", "d")
        fired = init(Rd=0.0, Rd2=0.0, Ld=300e-12)
        assert tuple(di_d) not in {tuple(p) for p in fired}, "(di,d) collapsed with only Ld>0"

    def test_dead_branches_collapse(self, cguard, init):
        _, di_d = _pair_by_names(cguard, "di", "d")
        _, si_s = _pair_by_names(cguard, "si", "s")
        fired = {tuple(p) for p in init()}  # every guard param 0
        assert tuple(di_d) in fired and tuple(si_s) in fired
        # extras fire with their sources -> every pair fires
        assert len(fired) == len(cguard.collapsible_pairs)

    def test_real_source_resistor_does_not_collapse(self, cguard, init):
        _, si_s = _pair_by_names(cguard, "si", "s")
        fired = init(Rs=3.7)
        assert tuple(si_s) not in {tuple(p) for p in fired}


class TestOldShapeBackCompat:
    """Old pickled caches carry (pair_idx, "vN"/"!vN") — normalized, not crashed on."""

    def test_normalize_guard(self):
        assert InitFunctionBuilder._normalize_guard((0, "!v5")) == (0, [("v5", True)])
        assert InitFunctionBuilder._normalize_guard((1, "v7")) == (1, [("v7", False)])
        # new shape passes through; JSON round-trips (lists, not tuples) normalize too
        assert InitFunctionBuilder._normalize_guard((2, [["v9", True], ("v4", False)])) == (
            2,
            [("v9", True), ("v4", False)],
        )
        assert InitFunctionBuilder._normalize_guard((3, [])) == (3, [])


class TestAsmhemtPin:
    """Regression pin for the model whose guards the index desync used to shift."""

    @pytest.fixture(scope="class")
    def asmhemt(self):
        if not ASMHEMT_VA.exists():
            pytest.skip(f"model not found: {ASMHEMT_VA}")
        return openvaf_py.compile_va(str(ASMHEMT_VA))[0]

    def test_every_pair_guarded(self, asmhemt):
        n = len(asmhemt.collapsible_pairs)
        assert n == 11
        guarded = {idx for idx, _ in asmhemt.collapse_decision_outputs}
        assert guarded == set(range(n)), (
            "pre-fix desync left pairs unguarded (the (g,gi) guard landed one pair low)"
        )

    def test_implicit_pair_single_plain_conjunct(self, asmhemt):
        # NodeCollapse::new inserts implicit-equation pairs FIRST, so pair 0 is
        # asmhemt's implicit_equation_0: a ground pair guarded by exactly one
        # entry with a single non-negated conjunct (its init output bool). The
        # pre-fix extraction left it unguarded and gave its slot a hint's guard.
        assert asmhemt.collapsible_pairs[0][1] == 4294967295
        guards = [conjs for idx, conjs in asmhemt.collapse_decision_outputs if idx == 0]
        assert len(guards) == 1
        assert len(guards[0]) == 1 and guards[0][0][1] is False
