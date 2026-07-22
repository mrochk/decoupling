import jax, jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import jaxtyped, Float, Array

from decoupling.result import Decoupling

class JacobianScaler:
    '''
    Rescale the Jacobians so that each output is equally minimized.\\
    Particularly useful for functions with vastly different gradient magnitudes.
    '''

    scaling_factors: Float[Array, 'n']

    @jaxtyped(typechecker=beartype)
    def __init__(self, jacobians: Float[Array, 'n m N']):
        n, m, N = jacobians.shape
        self.scaling_factors = jnp.linalg.norm(jacobians.reshape(n, -1), axis=1) / jnp.sqrt(m*N)
        
    @jaxtyped(typechecker=beartype)
    def scale(self, jacobians, outputs) -> Tuple[Float[Array, 'n m N'], Float[Array, 'N n']]:
        J_scaled = jacobians / self.scaling_factors[:, None, None]
        Y_scaled = outputs / self.scaling_factors[None, :]
        return J_scaled, Y_scaled

    def unscale(self, decoupling: Decoupling) -> Decoupling:
        W_unscaled = decoupling.W * self.scaling_factors[:, None]
        factors = (W_unscaled, decoupling.V, decoupling.H) + ((decoupling.R,) if decoupling.R is not None else ())
        return Decoupling(factors, decoupling.coefs, decoupling.knots, decoupling.degree)
