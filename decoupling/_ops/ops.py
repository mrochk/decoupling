import jax, jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import jaxtyped, Float, Array, ArrayLike

from decoupling.types import W_dtype, V_dtype, H_dtype, J_dtype

@jaxtyped(typechecker=beartype)
def convert_array(array: ArrayLike) -> Array:
    ''' convert an array-like to jax float32 array if it's not already '''
    return jnp.asarray(array, dtype=jnp.result_type(array, jnp.float32))

@jax.jit(static_argnames='mode')
@jaxtyped(typechecker=beartype)
def unfold_kolda(tensor: Array, mode: int) -> Array:
    ''' tensor unfolding as defined by Kolda & Bader '''
    return jnp.reshape(jnp.moveaxis(tensor, mode, 0), shape=(tensor.shape[mode], -1), order='F')

@jax.jit
@jaxtyped(typechecker=beartype)
def khatri_rao(A: Float[Array, 'm k'], B: Float[Array, 'n k']) -> Float[Array, 'm*n k']:
    ''' khatri-rao product of two matrices '''
    (m, k), (n, _) = A.shape, B.shape
    return (A[:, None, :] * B[None, :, :]).reshape(m*n, k)

@jax.jit
@jaxtyped(typechecker=beartype)
def reconstruct(W: W_dtype, V: V_dtype, H: H_dtype, weights: Float[Array, 'r']) -> J_dtype:
    ''' reconstruct full tensor from factors '''

    def forloop(r, tensor):
        weight = weights[r]
        w = W[:, r][:, None, None]
        v = V[:, r][None, :, None]
        h = H[:, r][None, None, :]
        rank1 = weight * w * v * h 
        return tensor + rank1

    p, m, n, rank = H.shape[0], V.shape[0], W.shape[0], W.shape[1]
    return jax.lax.fori_loop(0, rank, forloop, jnp.zeros(shape=(n, m, p)))

@jax.jit
@jaxtyped(typechecker=beartype)
def lstsq(X: Array, Y: Array) -> Tuple[Array, Float[Array, '']]:
    ''' wrapper around jnp.linalg.lsqtsq, returns solution and rcond '''
    solution, _, _, svalues = jnp.linalg.lstsq(X, Y)
    rcond = svalues[-1] / svalues[0]
    return solution.T, rcond

@jax.jit
@jaxtyped(typechecker=beartype)
def cmtf_lstsq(X1, X2, Y1, Y2, gamma: Float[Array, '']) -> Tuple[Array, Float[Array, '']]:
    gamma = jnp.sqrt(gamma)
    X = jnp.concatenate([X1, gamma*X2], axis=0)
    Y = jnp.concatenate([Y1, gamma*Y2], axis=0)
    return lstsq(X, Y)

@jax.jit
@jaxtyped(typechecker=beartype)
def normalize_columns_V(W: W_dtype, V: V_dtype) -> Tuple[W_dtype, V_dtype]:
    rank = W.shape[1]

    def _(i, W_V):
        W, V = W_V
        colV, colW = V[:, i], W[:, i]
        norm = jnp.linalg.norm(colV) + 1e-12
        V = V.at[:, i].set(colV / norm)
        W = W.at[:, i].set(colW * norm)
        return W, V

    return jax.lax.fori_loop(0, rank, _, (W, V))

@jax.jit(static_argnames='n')
@jaxtyped(typechecker=beartype)
def second_difference_matrix(n: int) -> Array:
    ''' return second-order finite difference matrix for computing P-splines coefs '''
    return jnp.diff(jnp.diff(jnp.eye(n), axis=0), axis=0)
