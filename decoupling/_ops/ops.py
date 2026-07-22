import jax, jax.numpy as jnp
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array, ArrayLike

def convert_array(array: ArrayLike) -> Array:
    return jnp.asarray(array, dtype=jnp.result_type(array, jnp.float32))

@jax.jit(static_argnames='mode')
def unfold_kolda(tensor: ArrayLike, mode: int) -> ArrayLike:
    return jnp.reshape(jnp.moveaxis(tensor, mode, 0), shape=(tensor.shape[mode], -1), order='F')

@jax.jit
@jaxtyped(typechecker=beartype)
def khatri_rao(A: Float[Array, 'm k'], B: Float[Array, 'n k']) -> Float[Array, 'mn k']:
    m, k = A.shape
    n, _ = B.shape
    return (A[:, None, :] * B[None, :, :]).reshape(m*n, k)

@jax.jit
@jaxtyped(typechecker=beartype)
def reconstruct(W: Float[Array, 'n r'], V: Float[Array, 'm r'], H: Float[Array, 'p r'], weights: Float[Array, 'r']) -> Float[Array, 'n m p']:
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
def lstsq(X, Y):
    solution, _, _, svalues = jnp.linalg.lstsq(X, Y)
    rcond = svalues[-1] / svalues[0]
    return solution.T, rcond

@jax.jit(static_argnames='gamma')
@jaxtyped(typechecker=beartype)
def cmtf_lstsq(X1, X2, Y1, Y2, gamma: float):
    gamma = jnp.sqrt(gamma)
    X = jnp.concatenate([X1, gamma*X2], axis=0)
    Y = jnp.concatenate([Y1, gamma*Y2], axis=0)
    return lstsq(X, Y)

@jax.jit
def normalize_columns_V(W: Float[Array, 'n r'], V: Float[Array, 'm r']):
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
def second_difference_matrix(n: int) -> Array:
    D1 = jnp.diff(jnp.eye(n), axis=0)
    D2 = jnp.diff(D1, axis=0)
    return D2
