import random
import jax, jax.numpy as jnp
from functools import partial
from bsplx import design_matrix, design_dmatrix

from jaxtyping import jaxtyped, Array, Float, ArrayLike
from beartype import beartype 
from beartype.typing import Callable, Union, Tuple

from decoupling import _ops as ops

def get_random_key() -> Array:
    return jax.random.key(random.randint(0, int(1e10)))

def find_number_inputs(function: Callable):
    assert callable(function)
    m = 1
    while True:
        try: function(jnp.zeros(m)); return m
        except (ValueError, TypeError): m += 1

### factors initialization

@jaxtyped(typechecker=beartype)
def initialize(tensor: Float[Array, 'n m N'], rank: int, key: Array, with_R: bool = False):
    n, m, N = tensor.shape
    keys = jax.random.split(key, num=4)

    W = jax.random.normal(keys[0], shape=(n, rank))
    V = jax.random.normal(keys[1], shape=(m, rank))
    H = jax.random.normal(keys[2], shape=(N, rank))

    if not with_R: return (W, V, H)

    R = jax.random.normal(keys[3], shape=(N, rank))
    return (W, V, H, R)

### CP decomposition

def cpd_stopping_criterion(diff: float, tol: float, norm: float) -> bool:
    return diff < tol * norm

@jaxtyped(typechecker=beartype)
def solve_cpd_subproblem(
    unfolded: Array, 
    W: Float[Array, 'n r'], 
    V: Float[Array, 'm r'],
    H: Float[Array, 'N r'], 
    mode: int,
):
    assert 0 <= mode <= 2
    match mode:
        case 0: return ops.cpd_factor_solve(unfolded, H, V)
        case 1: return ops.cpd_factor_solve(unfolded, H, W)
        case 2: return ops.cpd_factor_solve(unfolded, V, W)

### stuff related to fitting internals

@jax.jit
def bspline_project(i, coefs, B, dB, H, R):
    H = H.at[:, i].set(dB @ coefs)
    R = R.at[:, i].set(B @ coefs)
    return (H, R)

def fit_internal_with_best_coefs(coefs, knots, degree):
    def g(x):
        B = get_design_matrix(jnp.atleast_1d(x), knots, degree)
        return jnp.squeeze(B @ coefs)
    return g

def fit_internals_with_best_coefs(coefs_list, knots_list, degree):
    internals = []
    for coefs, knots in zip(coefs_list, knots_list):
        if coefs is None:
            internals.append(lambda x: jnp.zeros_like(x))
            continue
        internals.append(fit_internal_with_best_coefs(coefs, knots, degree))
    return internals

def get_design_matrix(z, knots: Array, degree: int):
    matrix = design_matrix(z, knots, degree)
    matrix = jnp.concatenate([jnp.ones((matrix.shape[0], 1)), matrix], axis=1)
    return matrix

def get_design_dmatrix(z, knots, degree: int):
    dmatrix = design_dmatrix(z, knots, degree)
    dmatrix = jnp.concatenate([jnp.zeros((dmatrix.shape[0], 1)), dmatrix], axis=1)
    return dmatrix

def determine_knots(z: Float[Array, 'r'], dof: int, degree: int, method: str) -> Array:
    match method:
        case 'even': return _determine_knots_even(z, dof, degree)
        case 'quantile': return _determine_knots_quantiles(z, dof, degree)
        case _: raise ValueError()

@jax.jit(static_argnames=('dof', 'degree'))
def _determine_knots_even(u: Float[Array, 'r'], dof: int, degree: int) -> Array:
    internals = dof - degree + 1

    knots = jnp.linspace(jnp.min(u), jnp.max(u), internals)

    begin = jnp.repeat(knots[0], degree)
    end   = jnp.repeat(knots[-1], degree)
    return jnp.concat([begin, knots, end])

@jax.jit(static_argnames=('dof', 'degree'))
def _determine_knots_quantiles(u: Float[Array, 'r'], dof: int, degree: int) -> Array:
    internals = dof - degree + 1

    qs = jnp.linspace(0, 1, internals)
    knots = jnp.quantile(u, qs)
    knots = jax.vmap(partial(_closest, u=u))(knots)

    begin = jnp.repeat(knots[0], degree)
    end = jnp.repeat(knots[-1], degree)
    return jnp.concat([begin, knots, end])

@jax.jit
def _closest(knot, u):
    def forloop(i, args):
        min_dist, closest_x = args
        x = u[i]
        dist = jnp.abs(x - knot)
        return jax.lax.cond(
            dist < min_dist,
            lambda: (dist, x),
            lambda: (min_dist, closest_x),
        )

    _, closest_point = jax.lax.fori_loop(0, len(u), forloop, (jnp.inf, u[0]))
    return closest_point

# hyperparameters

def default_dof(N):
    return max(min([2*int(jnp.sqrt(N))+1, N//2]), 1)

# dtype

def as_float_array(array: ArrayLike) -> Array:
    return jnp.asarray(array, dtype=jnp.result_type(array, jnp.float32))

dtype_factors = Union[
    Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'N r']],
    Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'N r'], Float[Array, 'N r']],
]
