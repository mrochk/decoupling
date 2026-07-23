import jax, jax.numpy as jnp
from beartype import beartype
from beartype.typing import Iterable
from jaxtyping import jaxtyped, Array

from decoupling.types import *
from decoupling._splines import design_matrix

class Decoupling:
    rank: int

    W: W_dtype 
    V: V_dtype
    H: H_dtype
    R: R_dtype
    factors: factors_dtype

    degree: int
    coefs: Array
    knots: Array

    unscaled_: bool = False

    @jaxtyped(typechecker=beartype)
    def __init__(self, factors: factors_dtype, coefs: Iterable, knots: Iterable, degree: int):
        assert len(factors) == 4

        (self.W, self.V, self.H, self.R) = factors

        self.factors = factors
        self.rank = self.W.shape[1]

        self.coefs = jnp.stack(coefs)
        self.knots = jnp.stack(knots)
        self.degree = degree

        @jax.jit
        def _single_internal_static(x, coefs, knots):
            dm = design_matrix(jnp.atleast_1d(x), knots, self.degree)
            return (dm @ coefs).squeeze()

        self._internals = jax.vmap(_single_internal_static)

    @staticmethod
    @jax.jit(static_argnames='internals')
    @jaxtyped(typechecker=beartype)
    def _forward(inputs: Float[Array, 'm'], W: W_dtype, internals, V: V_dtype):
        return W @ internals(V.T @ inputs)

    def forward(self, inputs: Float[Array, 'm']) -> Float[Array, 'n']:
        return self._forward(inputs, self.W, self.internals, self.V)

    def __call__(self, inputs: Float[Array, 'm']):
        return self.forward(inputs)

    @jaxtyped(typechecker=beartype)
    def internals(self, inputs: Float[Array, 'r']) -> Float[Array, 'r']:
        return self._internals(inputs, self.coefs, self.knots)
