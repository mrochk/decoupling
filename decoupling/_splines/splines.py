import bsplx
import jax, jax.numpy as jnp
from beartype import beartype
from bsplx import repeat_knots
from beartype.typing import Tuple
from jaxtyping import jaxtyped, Array, Float

from decoupling.types import *

@jax.jit(static_argnames='degree')
@jaxtyped(typechecker=beartype)
def design_matrix(inputs: Float[Array, 'n'], knots: Float[Array, 'k'], degree: int) -> Float[Array, 'n m']: # m = k - degree
    '''Get the B-spline design matrix with an added column of ones.'''
    matrix = bsplx.design_matrix(inputs, knots, degree)
    matrix = jnp.concatenate([jnp.ones((matrix.shape[0], 1)), matrix], axis=1)
    return matrix

@jax.jit(static_argnames='degree')
@jaxtyped(typechecker=beartype)
def design_dmatrix(inputs: Float[Array, 'n'], knots: Float[Array, 'k'], degree: int) -> Float[Array, 'n m']: # m = k - degree
    '''Get the B-spline derivative design matrix with an added column of zeros.'''
    dmatrix = bsplx.design_dmatrix(inputs, knots, degree)
    dmatrix = jnp.concatenate([jnp.zeros((dmatrix.shape[0], 1)), dmatrix], axis=1)
    return dmatrix

@jax.jit(static_argnames='degree')
@jaxtyped(typechecker=beartype)
def design_matrices(inputs: Float[Array, 'n'], knots: Float[Array, 'k'], degree: int) -> Tuple:
    return (design_matrix(inputs, knots, degree), design_dmatrix(inputs, knots, degree))

@staticmethod
@jax.jit(static_argnames='rank')
@jaxtyped(typechecker=beartype)
def project(rank: int, coefs: Array, B: Array, dB: Array, H: H_dtype, R: R_dtype) -> Tuple[H_dtype, R_dtype]:
    H_new = H.at[:, rank].set(dB @ coefs)
    R_new = R.at[:, rank].set(B @ coefs)
    return (H_new, R_new)

@staticmethod
@jax.jit
def gcv_demmler_reinsch(A: Array, y: Array, D: Array, grid: Array, N: Array) -> Tuple[Array, Array]:
    n, _ = A.shape

    M = jnp.concatenate([A, D], axis=0) # invertible matrix that contains D

    Qm, Rm = jnp.linalg.qr(M)
    Um, s, Vmt = jnp.linalg.svd(Rm)

    tol = jnp.finfo(A.dtype).eps * max(M.shape) * s[0]
    keep = s > tol
    inv_s = jnp.where(keep, 1.0 / jnp.where(keep, s, 1.0), 0.0)

    X = Qm[:n] @ jnp.where(keep[None, :], Um, 0.0)

    Qx, Rx = jnp.linalg.qr(X)
    Ux, c, Wt = jnp.linalg.svd(Rx)

    f = Ux.T @ (Qx.T @ y)
    r0 = y - Qx @ (Ux @ f)
    rss0 = r0 @ r0

    c2 = jnp.clip(c * c, 0.0, 1.0)
    d2 = 1.0 - c2

    lam = grid[:, None]
    denom = c2 + lam * d2
    rss = rss0 + jnp.sum((f * (lam * d2) / denom) ** 2, axis=1)
    dof = jnp.sum(c2 / denom, axis=1)

    scores = (rss / N) / (1.0 - dof / N) ** 2
    scores = jnp.where(jnp.isfinite(scores) & (dof < N), scores, jnp.inf)

    best = jnp.argmin(scores)
    lam = grid[best]

    a = c * f / denom[best]
    coefs = (Vmt.T * inv_s) @ (Wt.T @ a)

    return (lam, coefs)

@staticmethod
@jax.jit(static_argnames=('dof', 'degree'))
def determine_knots(z, dof: int, degree: int) -> Array:
    internals = dof - degree + 1
    qs = jnp.linspace(0, 1, internals)
    return repeat_knots(jnp.quantile(z, qs), degree)
