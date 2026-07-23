import bsplx
import jax, jax.numpy as jnp
from jaxtyping import jaxtyped, Array, Float, Int
from beartype import beartype

@jax.jit(static_argnames='degree')
@jaxtyped(typechecker=beartype)
def design_matrix(inputs: Float[Array, 'n'], knots: Float[Array, 'k'], degree: int) -> Float[Array, 'n m']: # m = k - degree
    ''' get the b-spline design matrix with an added column of ones '''
    matrix = bsplx.design_matrix(inputs, knots, degree)
    matrix = jnp.concatenate([jnp.ones((matrix.shape[0], 1)), matrix], axis=1)
    return matrix

@jax.jit(static_argnames='degree')
@jaxtyped(typechecker=beartype)
def design_dmatrix(inputs: Float[Array, 'n'], knots: Float[Array, 'k'], degree: int) -> Float[Array, 'n m']: # m = k - degree
    ''' get the b-spline derivative design matrix with an added column of zeros '''
    dmatrix = bsplx.design_dmatrix(inputs, knots, degree)
    dmatrix = jnp.concatenate([jnp.zeros((dmatrix.shape[0], 1)), dmatrix], axis=1)
    return dmatrix
