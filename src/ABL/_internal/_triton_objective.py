"""Fused Triton kernel for the stimulus optimizer's objective, callable from JAX.

Optional fast path for `StimulusOptimizer` on CUDA GPUs; requires `triton` and `jax-triton`.
It computes the same fitness as the XLA objective in `optimizer._create_objective_function`
(expected change in source probability, E-field limit penalty, optional strain penalty), but
in a single kernel: every per-(candidate, vertex) value stays in registers instead of being
written to and re-read from GPU memory, which is what limits the XLA version.

Each program handles BLOCK_P candidates and loops twice over the vertices in chunks of BLOCK_V:
  pass 1: online logsumexp of both outcome numerators (giving log P(y=1), log P(y=0)) and the
          maximum E-field magnitude;
  pass 2: the expected absolute change in source probability, recomputing the numerators.
"""

from typing import Callable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax_triton as jt
import numpy as np
import triton
import triton.language as tl

Array = jnp.ndarray

# Constraint types supported inside the kernel (selected at compile time).
STRAIN_NONE = 0
STRAIN_5COIL = 1

# Candidate block configurations for the autotuner. The best one depends on the GPU
# (e.g. 8 x 128 with 4 warps on an RTX 4060 Ti, 2 x 512 with 8 warps on an MI300X).
_CONFIGS = [
    triton.Config({'BLOCK_P': bp, 'BLOCK_V': bv}, num_warps=w)
    for bp, bv, w in [(1, 512, 8), (2, 256, 4), (2, 512, 8), (4, 128, 4), (8, 128, 4), (8, 128, 8), (16, 128, 8)]
]

# Order of the response function parameters in the kernel's parameter array
_KERNEL_PARAM_ORDER = ('rf_slope', 'rf_translate', 'rf_fourier1', 'rf_fourier2', 'rf_fourier3', 'rf_fourier4')


@triton.jit
def _row(ptr, r, V, v_offs, v_mask):
    """Row r of a [R, V] array for this vertex chunk, shaped [1, BLOCK_V] for broadcasting."""
    return tl.load(ptr + r * V + v_offs, mask=v_mask, other=0.0)[None, :]


@triton.jit
def _tile(pop_ptr, p_offs, p_mask, W, B, CS, PRM, LSP, V, v_offs, v_mask, lo, hi,
          N_COILS: tl.constexpr, BLOCK_P: tl.constexpr, BLOCK_V: tl.constexpr):
    """Numerators for y=1 and y=0, E-field magnitude and log-prior on a [BLOCK_P, BLOCK_V] tile."""
    # E-field of every candidate at every vertex of the chunk: weighted sum over the coils
    E0 = tl.zeros([BLOCK_P, BLOCK_V], dtype=tl.float32)
    E1 = tl.zeros([BLOCK_P, BLOCK_V], dtype=tl.float32)
    E2 = tl.zeros([BLOCK_P, BLOCK_V], dtype=tl.float32)
    for c in tl.static_range(N_COILS):
        w = tl.load(pop_ptr + p_offs * N_COILS + c, mask=p_mask, other=0.0)[:, None]
        E0 += w * _row(W, c * 3 + 0, V, v_offs, v_mask)
        E1 += w * _row(W, c * 3 + 1, V, v_offs, v_mask)
        E2 += w * _row(W, c * 3 + 2, V, v_offs, v_mask)
    e_mag = tl.sqrt(E0 * E0 + E1 * E1 + E2 * E2)

    # Projection to the 2D subspace and its direction (cos phi, sin phi)
    t0 = E0 * _row(B, 0, V, v_offs, v_mask) + E1 * _row(B, 2, V, v_offs, v_mask) + E2 * _row(B, 4, V, v_offs, v_mask)
    t1 = E0 * _row(B, 1, V, v_offs, v_mask) + E1 * _row(B, 3, V, v_offs, v_mask) + E2 * _row(B, 5, V, v_offs, v_mask)
    inv = 1.0 / tl.sqrt(t0 * t0 + t1 * t1)
    cphi = t0 * inv
    sphi = t1 * inv

    # theta = phi - phi0 without atan2: angle-difference and double-angle identities
    ca = _row(CS, 0, V, v_offs, v_mask)
    sa = _row(CS, 1, V, v_offs, v_mask)
    cos1 = cphi * ca + sphi * sa
    sin1 = sphi * ca - cphi * sa
    cos2 = cos1 * cos1 - sin1 * sin1
    sin2 = 2.0 * sin1 * cos1

    # Response model (directional sigmoid)
    trans = (_row(PRM, 1, V, v_offs, v_mask) + _row(PRM, 2, V, v_offs, v_mask) * cos1 + _row(PRM, 3, V, v_offs, v_mask) * sin1
             + _row(PRM, 4, V, v_offs, v_mask) * cos2 + _row(PRM, 5, V, v_offs, v_mask) * sin2)
    inner = tl.clamp(-_row(PRM, 0, V, v_offs, v_mask) * (e_mag - trans), -16.0, 16.0)
    r = tl.clamp(e_mag / (e_mag + tl.exp(inner)), lo, hi)

    lsp = _row(LSP, 0, V, v_offs, v_mask)
    num1 = lsp + tl.log(r)            # log prior + log P(y=1)
    num0 = lsp + tl.log(1.0 - r)      # log prior + log P(y=0)
    return num1, num0, e_mag, lsp


@triton.autotune(configs=_CONFIGS, key=['P', 'V'])
@triton.jit
def _objective_kernel(pop_ptr, W, B, CS, PRM, LSP, P, V, fit_ptr,
                      lo: tl.constexpr, hi: tl.constexpr, efield_limit: tl.constexpr, penalty: tl.constexpr,
                      strain_limit: tl.constexpr, N_COILS: tl.constexpr, STRAIN_MODE: tl.constexpr,
                      BLOCK_P: tl.constexpr, BLOCK_V: tl.constexpr):
    pid = tl.program_id(axis=0)
    p_offs = pid * BLOCK_P + tl.arange(0, BLOCK_P)
    p_mask = p_offs < P

    # Pass 1: online logsumexp over vertices for both outcomes, and the max E-field
    m1 = tl.full([BLOCK_P], float("-inf"), tl.float32)
    s1 = tl.zeros([BLOCK_P], dtype=tl.float32)
    m0 = tl.full([BLOCK_P], float("-inf"), tl.float32)
    s0 = tl.zeros([BLOCK_P], dtype=tl.float32)
    emax = tl.zeros([BLOCK_P], dtype=tl.float32)
    for start in range(0, V, BLOCK_V):
        v_offs = start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < V
        num1, num0, e_mag, lsp = _tile(pop_ptr, p_offs, p_mask, W, B, CS, PRM, LSP, V, v_offs, v_mask, lo, hi,
                                       N_COILS, BLOCK_P, BLOCK_V)
        vm = v_mask[None, :]
        num1 = tl.where(vm, num1, float("-inf"))
        num0 = tl.where(vm, num0, float("-inf"))
        emax = tl.maximum(emax, tl.max(tl.where(vm, e_mag, 0.0), axis=1))
        m1_new = tl.maximum(m1, tl.max(num1, axis=1))
        s1 = s1 * tl.exp(m1 - m1_new) + tl.sum(tl.exp(num1 - m1_new[:, None]), axis=1)
        m1 = m1_new
        m0_new = tl.maximum(m0, tl.max(num0, axis=1))
        s0 = s0 * tl.exp(m0 - m0_new) + tl.sum(tl.exp(num0 - m0_new[:, None]), axis=1)
        m0 = m0_new
    lse1 = m1 + tl.log(s1)            # log P(y=1) per candidate
    lse0 = m0 + tl.log(s0)
    py1 = tl.exp(lse1)

    # Pass 2: expected absolute change in source probability, numerators recomputed
    acc = tl.zeros([BLOCK_P], dtype=tl.float32)
    for start in range(0, V, BLOCK_V):
        v_offs = start + tl.arange(0, BLOCK_V)
        v_mask = v_offs < V
        num1, num0, e_mag, lsp = _tile(pop_ptr, p_offs, p_mask, W, B, CS, PRM, LSP, V, v_offs, v_mask, lo, hi,
                                       N_COILS, BLOCK_P, BLOCK_V)
        src = tl.exp(lsp)
        term = ((1.0 - py1)[:, None] * tl.abs(src - tl.exp(num0 - lse0[:, None]))
                + py1[:, None] * tl.abs(src - tl.exp(num1 - lse1[:, None])))
        acc += tl.sum(tl.where(v_mask[None, :], term, 0.0), axis=1)

    fit = -acc / V
    viol = tl.maximum(emax - efield_limit, 0.0)
    fit += penalty * viol * viol
    if STRAIN_MODE == 1:
        # 5-coil strain penalty, port of optimizer_constraints.get_5coil_strain_constraints
        a0 = tl.abs(tl.load(pop_ptr + p_offs * N_COILS + 0, mask=p_mask, other=0.0))
        a1 = tl.abs(tl.load(pop_ptr + p_offs * N_COILS + 1, mask=p_mask, other=0.0))
        a2 = tl.abs(tl.load(pop_ptr + p_offs * N_COILS + 2, mask=p_mask, other=0.0))
        a3 = tl.abs(tl.load(pop_ptr + p_offs * N_COILS + 3, mask=p_mask, other=0.0))
        a4 = tl.abs(tl.load(pop_ptr + p_offs * N_COILS + 4, mask=p_mask, other=0.0))
        coil5 = (tl.sqrt(0.9 * a0 * a0 + a1 * a1) * 0.2 + tl.sqrt(0.9 * a2 * a2 + a3 * a3)) * a4
        coil1 = (0.9 * a1 + 0.7 * tl.sqrt(a2 * a2 + 0.9 * a3 * a3) + 0.2 * a4) * a0
        sv = tl.maximum(tl.maximum(coil5, coil1) - strain_limit, 0.0)
        fit += penalty * sv * sv
    tl.store(fit_ptr + p_offs, fit, mask=p_mask)


def create_triton_objective_function(
    efield_basis: np.ndarray,
    null_dir_basis: np.ndarray,
    prob_lim: Sequence[float],
    efield_set: np.ndarray,
    max_current_slope: np.ndarray,
    efield_limit: float,
    strain_mode: int,
    strain_limit: float,
    constraint_penalty: float,
    param_keys: List[str],
    jax_device: jax.Device,
) -> Callable[[Array, Array, Tuple], Array]:
    """Returns a jitted objective with the same signature and result as the XLA one.

    Args:
        efield_basis: PCA basis of the E-field, shape (n_vertices, 3, 2).
        null_dir_basis: Reference direction in the 2D subspace, shape (n_vertices, 2).
        prob_lim: Min/max limits for clamping response probabilities.
        efield_set: E-field maps of all coils, shape (n_coils, n_vertices, 3).
        max_current_slope: Maximum dI/dt per coil, shape (n_coils,).
        efield_limit: Maximum allowed E-field strength.
        strain_mode: STRAIN_NONE or STRAIN_5COIL.
        strain_limit: Strain limit (ignored for STRAIN_NONE).
        constraint_penalty: Penalty weight for constraint violations.
        param_keys: Order of the response model parameters in `param_tuple`.
        jax_device: Device on which the constant arrays are placed.
    """
    n_coils, n_vertices, _ = efield_set.shape
    if strain_mode == STRAIN_5COIL and n_coils != 5:
        raise ValueError("The 5-coil strain constraint requires exactly 5 coils.")

    # Per-session constants, reordered so every row is contiguous along vertices (coalesced loads)
    efield_norm = efield_set.astype(np.float32) * max_current_slope.astype(np.float32).reshape(-1, 1, 1)
    W = np.ascontiguousarray(efield_norm.transpose(0, 2, 1).reshape(n_coils * 3, n_vertices))  # row c*3+d
    B = np.ascontiguousarray(efield_basis.astype(np.float32).transpose(1, 2, 0).reshape(6, n_vertices))  # row d*2+t
    a0 = np.arctan2(null_dir_basis[:, 1], null_dir_basis[:, 0]).astype(np.float32)
    CS = np.ascontiguousarray(np.stack([np.cos(a0), np.sin(a0)]))
    W, B, CS = (jax.device_put(x, jax_device) for x in (W, B, CS))
    prm_index = [param_keys.index(k) for k in _KERNEL_PARAM_ORDER]

    constexprs = dict(lo=float(prob_lim[0]), hi=float(prob_lim[1]), efield_limit=float(efield_limit),
                      penalty=float(constraint_penalty),
                      strain_limit=float(strain_limit) if strain_mode == STRAIN_5COIL else 0.0,
                      N_COILS=int(n_coils), STRAIN_MODE=int(strain_mode))

    def objective(population: Array, log_source_prob: Array, param_tuple: Tuple) -> Array:
        n_pop = population.shape[0]
        prm = jnp.stack([param_tuple[i] for i in prm_index]).astype(jnp.float32)
        return jt.triton_call(
            population.astype(jnp.float32), W, B, CS, prm, log_source_prob.astype(jnp.float32),
            n_pop, n_vertices,
            kernel=_objective_kernel,
            out_type=jax.ShapeDtypeStruct((n_pop,), jnp.float32),
            grid=lambda meta: (triton.cdiv(n_pop, meta['BLOCK_P']),),
            **constexprs,
        )

    return jax.jit(objective)
