import jax, jax.numpy as jnp
from jaxtyping import jaxtyped, Float, Array, ArrayLike
from beartype.typing import Iterable, Tuple
from beartype import beartype

@jax.jit(static_argnames=('shape',))
def reshape(tensor: Array, shape: int | Tuple[int]):
    return jnp.reshape(tensor, shape, order='F')

@jax.jit(static_argnames=('mode',))
def unfold_kolda(tensor: ArrayLike, mode: int) -> ArrayLike:
    '''Tensor unfolding as defined in "Tensor decompositions and applications" from Kolda and Bader.'''
    return reshape(jnp.moveaxis(tensor, mode, 0), shape=(tensor.shape[mode], -1))

@jax.jit
@jaxtyped(typechecker=beartype)
def khatri_rao(A: Float[Array, 'm k'], B: Float[Array, 'n k']) -> Float[Array, 'mn k']:
    m, k = A.shape
    n, _ = B.shape
    return (A[:, None, :] * B[None, :, :]).reshape(m*n, k)

@jax.jit
def cpd_factor_solve(unfolded, A, B):
    KR = khatri_rao(A, B)
    CC = A.T @ A
    BB = B.T @ B
    return unfolded @ KR @ jnp.linalg.pinv(CC * BB)

@jax.jit
def block_diag(arrays: Iterable[ArrayLike]) -> Float[Array, 'a b']:
    arrays = [jnp.atleast_2d(a) for a in arrays]
    rows = sum([a.shape[0] for a in arrays])
    cols = sum([a.shape[1] for a in arrays])

    result = jnp.zeros((rows, cols), dtype=arrays[0].dtype)

    r, c = 0, 0
    for a in arrays:
        rr, cc = a.shape
        result = result.at[r:r+rr, c:c+cc].set(a)
        r += rr
        c += cc

    return result

@jax.jit
def reconstruct(W: Array, V: Array, H: Array, weights: Array) -> Array:
    def forloop(r, tensor):
        weight = weights[r]
        w = W[:, r][:, None, None]
        v = V[:, r][None, :, None]
        h = H[:, r][None, None, :]
        rank1 = weight * w * v * h 
        return tensor + rank1

    N, m, n, rank = H.shape[0], V.shape[0], W.shape[0], W.shape[1]
    return jax.lax.fori_loop(0, rank, forloop, jnp.zeros(shape=(n, m, N)))

### vandermonde stuff

def vandermonde_vector(x: float, d: int):
    return jnp.array([x**e for e in range(d + 1)])

def vandermonde_matrix(values: Iterable[float], degree: int):
    return jnp.vstack([vandermonde_vector(v, degree) for v in values])

def vandermonde_diag(X, d: int):
    return block_diag([vandermonde_vector(x, d) for x in X])

### wrappers for least squares funcs

@jax.jit
def lstsq(X, Y): return jnp.linalg.lstsq(X, Y)[0].T

@jax.jit(static_argnames='gamma')
@jaxtyped(typechecker=beartype)
def cmtf_lstsq(X1, X2, Y1, Y2, gamma: float):
    X = jnp.concatenate([X1, gamma*X2], axis=0)
    Y = jnp.concatenate([Y1, gamma*Y2], axis=0)
    return jnp.linalg.lstsq(X, Y)[0].T

### normalization

@jax.jit
def normalize_columns_simple(factor: Float[Array, '_ r']) -> Tuple[Float[Array, '_ r'], Float[Array, 'r']]:
    rank = factor.shape[1]
    weights = jnp.empty(rank)

    def forloop(r, factor_weights):
        factor, weights = factor_weights
        column = factor[:, r]
        norm = jnp.linalg.norm(column)
        weights = weights.at[r].set(norm)
        factor = factor.at[:, r].set(column / norm)
        return factor, weights

    return jax.lax.fori_loop(0, rank, forloop, (factor, weights))

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
def second_diff_matrix(n: int) -> Array:
    D1 = jnp.diff(jnp.eye(n), axis=0)
    D2 = jnp.diff(D1, axis=0)
    return D2
