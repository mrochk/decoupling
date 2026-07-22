import random
import jax, jax.numpy as jnp
from functools import partial
from bsplx import design_matrix, design_dmatrix

from jaxtyping import Array, Float, ArrayLike
from beartype.typing import Callable

def get_random_key() -> Array:
    return jax.random.key(random.randint(0, int(1e10)))

def find_ninputs(function: Callable):
    assert callable(function)
    m = 1
    while True:
        try: function(jnp.zeros(m)); return m
        except (ValueError, TypeError): m += 1

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

def as_float_array(array: ArrayLike) -> Array:
    return jnp.asarray(array, dtype=jnp.result_type(array, jnp.float32))
