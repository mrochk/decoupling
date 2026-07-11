from jaxtyping import jaxtyped, ArrayLike, Float
from beartype.typing import Callable
from beartype import beartype

from decoupling._common import dtype_factors

class Decoupling:

    factors: dtype_factors
    internals: Callable

    @jaxtyped(typechecker=beartype)
    def __init__(self, factors: dtype_factors, internals: Callable):
        assert len(factors) >= 3
        assert callable(internals)
        assert factors[0].shape[-1] == factors[1].shape[-1] == factors[2].shape[-1]

        self.factors = factors
        if len(factors) == 4: self.W, self.V, self.H, self.R = factors
        else: self.W, self.V, self.H = factors
        self.internals = internals

    @jaxtyped(typechecker=beartype)
    def __call__(self, x: Float[ArrayLike, 'm']) -> Float[ArrayLike, 'n']:
        return self.W @ self.internals(self.V.T @ x)

    def inference(self, x: ArrayLike) -> ArrayLike: return self.__call__(x)
