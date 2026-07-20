import jax, jax.numpy as jnp
from jaxtyping import jaxtyped, Array, ArrayLike, Float
from beartype import beartype
from beartype.typing import Tuple, Optional

from decoupling._common import as_float_array, get_random_key, dtype_factors
from decoupling.result.result import _Decoupling

class _Base:
    '''Base class for all tensor decoupling algorithms.'''

    rank: int
    niters: int # max number of iterations
    ninits: int # number of random seeds tried
    key: Array  # base random seed
    show_progress: bool
    errors: Array = jnp.empty(0)

    @jaxtyped(typechecker=beartype)
    def __init__(self, rank: int, niters: int, ninits: int, key: Optional[Array], show_progress: bool):
        self.rank = rank
        self.niters = niters
        self.ninits = ninits
        self.show_progress = show_progress
        self.key = get_random_key() if key is None else key 

    @jaxtyped(typechecker=beartype)
    def _toarrays(
        self,
        inputs: Float[ArrayLike, 'N m'],
        outputs: Float[ArrayLike, 'N n'],
        jacobians: Float[ArrayLike, 'n m N'],
    ) -> Tuple[Float[Array, 'N m'], Float[Array, 'N n'], Float[Array, 'n m N']]:
        return (
            as_float_array(inputs), 
            as_float_array(outputs), 
            as_float_array(jacobians),
        )

    def run(
        self,
        inputs: Float[ArrayLike, 'N m'],
        outputs: Float[ArrayLike, 'N n'],
        jacobians: Float[ArrayLike, 'n m N'],
    ) -> _Decoupling: pass

    @jaxtyped(typechecker=beartype)
    def _initialize_factors(self, jacobians: Float[Array, 'n m N'], key: Array, with_R: bool = False) -> dtype_factors:
        n, m, N = jacobians.shape
        keys = jax.random.split(key, num=4)

        W = jax.random.normal(keys[0], shape=(n, self.rank))
        V = jax.random.normal(keys[1], shape=(m, self.rank))
        H = jax.random.normal(keys[2], shape=(N, self.rank))

        if not with_R: return (W, V, H)

        R = jax.random.normal(keys[3], shape=(N, self.rank))
        return (W, V, H, R)
