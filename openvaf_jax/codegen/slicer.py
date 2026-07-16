"""Statement-level backward slicing + sequential segmentation of the generated eval.

VASAX 2nd-order ∂jac/∂param support (KB §3q). The feature-on eval is a single flat
generated function whose XLA/LLVM compile is super-linear in size (BSIM4 w,l: 37k ops,
>2 h). This module slices the emitted statement list into

  * a **value body** — the original 14 outputs with the two ``jacobian_*_dparam``
    slots replaced by shape ``(n_entries, 0)`` empties, so the value eval shrinks
    back to (near) feature-off size and the load-time warmup stays fast; and
  * **K chained dparam segment functions** — the backward-slice closure of the two
    dparam outputs, cut into ~budget-sized sequential segments that pass their
    boundary-live values as a carry tuple. Each segment is emitted as its own
    module and jitted separately, so LLVM never sees the monolith (measured:
    BSIM4 w,l 6×~5.7k stmts → ~40 s total vs >2 h; chained segments match the
    monolith exactly).

Statement model: the emitted body is a flat list of assignments (plus leading
imports, loop-helper ``FunctionDef``s and their ``lax.while_loop``/``lax.scan``
result assignment, and one trailing ``Return``). Def/use is computed per
statement; reverse liveness handles pre-init/re-assignment (kill on def). Loop
helper defs bind their name and use their free variables, so slices keep them
exactly when the loop result is live; segment cuts are forbidden between a
``FunctionDef`` and the statement consuming it (a function cannot be carried
through jit).
"""

import ast
import os
from typing import Dict, List, Sequence, Set, Tuple

DPARAM_OUTS = ("jacobian_resist_dparam", "jacobian_react_dparam")

# Per-segment statement budget. ~6k statements compile in 0.5-3 s each on CPU
# (measured on the real BSIM4/EKV feature-on evals); the LLVM blow-up starts
# well above ~12k. Overridable for experiments.
_DEFAULT_BUDGET = int(os.environ.get("OPENVAF_2ND_ORDER_SEG_STMTS", "6000"))


class _Names(ast.NodeVisitor):
    def __init__(self) -> None:
        self.loads: Set[str] = set()
        self.stores: Set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        (self.loads if isinstance(node.ctx, ast.Load) else self.stores).add(node.id)


def _defs_uses(stmt: ast.stmt) -> Tuple[Set[str], Set[str]]:
    """(defined names, used/free names) of one top-level statement."""
    if isinstance(stmt, (ast.Import, ast.ImportFrom)):
        return {a.asname or a.name.split(".")[0] for a in stmt.names}, set()
    if isinstance(stmt, ast.FunctionDef):
        v = _Names()
        for s in stmt.body:
            v.visit(s)
        bound = {a.arg for a in stmt.args.args} | v.stores
        return {stmt.name}, v.loads - bound
    v = _Names()
    v.visit(stmt)
    return v.stores, v.loads - v.stores


def _backward_slice(
    body: Sequence[ast.stmt],
    live: Set[str],
    du: Sequence[Tuple[Set[str], Set[str]]],
) -> List[int]:
    """Indices of statements kept by reverse liveness from ``live`` (imports always kept)."""
    live = set(live)
    kept: List[int] = []
    for i in range(len(body) - 1, -1, -1):
        d, _u = du[i]
        if d & live or isinstance(body[i], (ast.Import, ast.ImportFrom)):
            kept.append(i)
            live -= d
            live |= du[i][1]
    kept.reverse()
    return kept


def slice_value_body(body: List[ast.stmt], n_jac_entries: int) -> List[ast.stmt]:
    """The value eval: original 14-tuple with EMPTY dparam slots, sliced to size.

    The two ``jacobian_*_dparam`` assignments are replaced by ``jnp.zeros((n, 0))``
    (the feature-off shape convention), then reverse liveness drops the tangent
    chains whose only consumers they were. Statement objects are shared with the
    original body (they are only re-parented into a new FunctionDef).
    """
    empty = ast.parse(f"_x = jnp.zeros(({n_jac_entries}, 0))").body[0].value
    out: List[ast.stmt] = []
    for s in body:
        if (
            isinstance(s, ast.Assign)
            and isinstance(s.targets[0], ast.Name)
            and s.targets[0].id in DPARAM_OUTS
        ):
            out.append(ast.Assign(targets=s.targets, value=empty))
        else:
            out.append(s)
    du = [_defs_uses(s) for s in out]
    ret = next(s for s in reversed(out) if isinstance(s, ast.Return))
    live = {e.id for e in ret.value.elts if isinstance(e, ast.Name)}
    keep = _backward_slice(out, live, du)
    # the Return defines nothing, so the slice drops it — re-append it explicitly
    return [out[i] for i in keep if not isinstance(out[i], ast.Return)] + [ret]


def _cut_points(sliced: Sequence[ast.stmt], budget: int) -> List[int]:
    """Segment bounds [0, ..., n] at ~budget strides, never splitting a loop group.

    A cut index p means "segment boundary before statement p". Cutting directly
    after a ``FunctionDef`` (loop helper) would strand it from its consuming
    ``lax.while_loop``/``lax.scan`` assignment — functions cannot cross a jit
    boundary — so such positions are pushed forward past the group.
    """
    n = len(sliced)
    k = max(1, round(n / budget))
    bounds = [round(n * i / k) for i in range(k + 1)]
    for j in range(1, len(bounds) - 1):
        p = bounds[j]
        # a cut at p is invalid while the previous statement is a FunctionDef
        # (helper defs directly precede their consuming assignment)
        while 0 < p < n and isinstance(sliced[p - 1], ast.FunctionDef):
            p += 1
        bounds[j] = p
    # de-duplicate / keep monotone (tiny bodies or pushed cuts may collide)
    out = [bounds[0]]
    for b in bounds[1:]:
        if b > out[-1]:
            out.append(b)
    if out[-1] != n:
        out.append(n)
    return out


def build_dparam_segments(
    body: List[ast.stmt],
    fn_args: Sequence[str],
    budget: int = 0,
) -> Tuple[List[Tuple[str, str]], Dict[str, object]]:
    """Emit the chained dparam segment modules from the full feature-on body.

    Returns ``(segments, meta)`` where ``segments`` is a list of
    ``(fn_name, module_source)`` — each segment its own self-contained module —
    and ``meta`` records the carry sizes and statement counts. Segment 0 takes the
    eval arguments; segments 1..K-1 take ``(carry, *eval_args)`` (carry first, so
    trailing kwargs like ``limit_funcs`` can be partial()'d away); the last segment
    returns ``(jacobian_resist_dparam, jacobian_react_dparam)``.
    """
    budget = budget or _DEFAULT_BUDGET
    du_all = [_defs_uses(s) for s in body]
    keep = [
        i
        for i in _backward_slice(body, set(DPARAM_OUTS), du_all)
        if not isinstance(body[i], ast.Return)
    ]
    sliced = [body[i] for i in keep]
    sdu = [du_all[i] for i in keep]
    n = len(sliced)
    bounds = _cut_points(sliced, budget)
    k = len(bounds) - 1

    # names that live at module scope in every emitted segment (never carried)
    outer = set(fn_args)
    imports: List[ast.stmt] = []
    for s in sliced:
        if isinstance(s, (ast.Import, ast.ImportFrom)):
            imports.append(s)
            outer |= _defs_uses(s)[0]

    # per-segment defs / first-uses
    seg_defs: List[Set[str]] = []
    seg_uses: List[Set[str]] = []
    for j in range(k):
        d: Set[str] = set()
        u: Set[str] = set()
        for i in range(bounds[j], bounds[j + 1]):
            u |= sdu[i][1] - d
            d |= sdu[i][0]
        seg_defs.append(d)
        seg_uses.append(u)
    carries: List[List[str]] = []  # carry INTO segment j (j >= 1)
    for j in range(1, k):
        needed: Set[str] = set(DPARAM_OUTS)
        for m in range(j, k):
            needed |= seg_uses[m]
        avail: Set[str] = set()
        for m in range(j):
            avail |= seg_defs[m]
        carries.append(sorted((needed & avail) - outer))

    def _name_tuple(names: Sequence[str], ctx: type) -> ast.Tuple:
        return ast.Tuple(elts=[ast.Name(id=x, ctx=ctx()) for x in names], ctx=ctx())

    segments: List[Tuple[str, str]] = []
    for j in range(k):
        cin = carries[j - 1] if j >= 1 else []
        cout = carries[j] if j < k - 1 else list(DPARAM_OUTS)
        # carry comes FIRST so consumers can partial() trailing kwargs (limit_funcs)
        # and still call the segments positionally
        arg_names = (["carry"] if cin else []) + list(fn_args)
        seg_body: List[ast.stmt] = []
        if cin:
            seg_body.append(
                ast.Assign(
                    targets=[_name_tuple(cin, ast.Store)],
                    value=ast.Name(id="carry", ctx=ast.Load()),
                )
            )
        for i in range(bounds[j], bounds[j + 1]):
            if not isinstance(sliced[i], (ast.Import, ast.ImportFrom)):
                seg_body.append(sliced[i])
        seg_body.append(ast.Return(value=_name_tuple(cout, ast.Load)))
        fn = ast.FunctionDef(
            name=f"eval_dparam_seg{j}",
            args=ast.arguments(
                posonlyargs=[],
                args=[ast.arg(arg=a) for a in arg_names],
                kwonlyargs=[],
                kw_defaults=[],
                defaults=[],
            ),
            body=seg_body,
            decorator_list=[],
        )
        module = ast.Module(
            body=ast.parse("import jax\nfrom jax import numpy as jnp, lax").body + [fn],
            type_ignores=[],
        )
        ast.fix_missing_locations(module)
        segments.append((f"eval_dparam_seg{j}", ast.unparse(module)))

    meta = {
        "n_segments": k,
        "closure_stmts": n,
        "segment_stmts": [bounds[j + 1] - bounds[j] for j in range(k)],
        "carry_sizes": [len(c) for c in carries],
    }
    return segments, meta
