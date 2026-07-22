import jax, jax.numpy as jnp
from beartype import beartype
from beartype.typing import Callable, Tuple, Optional
from jaxtyping import jaxtyped, Float, Array, ArrayLike

from decoupling.utils import get_random_key
from decoupling._common import find_ninputs

class JacobianScaler:
    '''
    Rescale the Jacobians so that each output is equally minimized.
    Particularly useful for functions with vastly different output (and gradient) ranges.
    '''

    factors: Float[Array, 'n']

    @jaxtyped(typechecker=beartype)
    def __init__(self, jacobians: Float[Array, 'n m N']):
        n, m, N = jacobians.shape
        self.factors = jnp.sqrt(m*N) / jnp.linalg.norm(jacobians.reshape(n, -1), axis=1)

    @classmethod
    def from_inputs(cls, target: Callable, inputs: Float[ArrayLike, 'N m']):
        assert callable(target)
        jacobians = jax.vmap(jax.jacobian(target))(inputs).transpose((1, 2, 0))
        return cls(jacobians)

    @classmethod
    def from_random(
        cls, 
        target: Callable, 
        N: int, 
        ninputs: Optional[int] = None,
        key: Optional[Array] = None, 
        minval: float = 0.0,
        maxval: float = 1.0,
        dist: str = 'uniform',
    ):
        assert callable(target)

        if ninputs is None: ninputs = find_ninputs(target)
        if key is None: key = get_random_key()

        match dist:
            case 'uniform': inputs = jax.random.uniform(key, (N, ninputs), minval=minval, maxval=maxval)
            case _: raise NotImplementedError()

        jacobians = jax.vmap(jax.jacobian(target))(inputs).transpose((1, 2, 0))
        return cls(jacobians)
        
    @jaxtyped(typechecker=beartype)
    def scale(self, jacobians, outputs) -> Tuple[Float[Array, 'n m N'], Float[Array, 'N n']]:
        J_scaled = jacobians * self.factors[:, None, None]
        Y_scaled = outputs * self.factors[None, :]
        return J_scaled, Y_scaled

    @jaxtyped(typechecker=beartype)
    def unscale_function(self, f_scaled: Callable) -> Callable:
        def f_unscaled(x): return f_scaled(x) / self.factors
        return jax.jit(f_unscaled)

    @jax.jit
    @jaxtyped(typechecker=beartype)
    def unscale_output(self, output: Float[Array, 'n']) -> Float[Array, 'n']:
        return output / self.factors
