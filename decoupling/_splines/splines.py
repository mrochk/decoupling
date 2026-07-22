import bsplx
import jax.numpy as jnp

def design_matrix(z, knots, degree: int):
    matrix = bsplx.design_matrix(z, knots, degree)
    matrix = jnp.concatenate([jnp.ones((matrix.shape[0], 1)), matrix], axis=1)
    return matrix

def design_dmatrix(z, knots, degree: int):
    dmatrix = bsplx.design_dmatrix(z, knots, degree)
    dmatrix = jnp.concatenate([jnp.zeros((dmatrix.shape[0], 1)), dmatrix], axis=1)
    return dmatrix
