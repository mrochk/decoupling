import jax, jax.numpy as jnp
from jaxtyping import jaxtyped, Array
from beartype import beartype
from beartype.typing import Optional

from decoupling.types import *
from decoupling._splines import design_matrix

class Decoupling:
    rank: int
    factors: factors_dtype
    W: W_dtype 
    V: V_dtype
    H: H_dtype
    R: Optional[R_dtype]

    degree: int
    coefs: Array
    knots: Array

    unscaled_: bool = False

    @jaxtyped(typechecker=beartype)
    def __init__(self, factors: factors_dtype, coefs, knots, degree: int):
        assert len(factors) >= 3
        assert factors[0].shape[-1] == factors[1].shape[-1] == factors[2].shape[-1]

        if len(factors) == 4: self.W, self.V, self.H, self.R = factors
        else: self.W, self.V, self.H = factors

        self.factors = factors
        self.rank = self.W.shape[1]

        self.coefs = jnp.stack(coefs)
        self.knots = jnp.stack(knots)
        self.degree = degree

        def _single_internal_static(x, coefs, knots):
            dm = design_matrix(jnp.atleast_1d(x), knots, self.degree)
            return (dm @ coefs).squeeze()

        self._internals = jax.vmap(_single_internal_static)

    @staticmethod
    @jax.jit(static_argnames='internals')
    def _forward(inputs, W, internals, V):
        return W @ internals(V.T @ inputs)

    def forward(self, inputs):
        return self._forward(inputs, self.W, self.internals, self.V)

    def __call__(self, inputs):
        return self.forward(inputs)

    def single_internal(self, x, idx: int):
        dm = design_matrix(jnp.atleast_1d(x), self.knots[idx], self.degree)
        return (dm @ self.coefs[idx]).squeeze()

    def internals(self, inputs):
        return self._internals(inputs, self.coefs, self.knots)
