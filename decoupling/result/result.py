import jax, jax.numpy as jnp
from jaxtyping import jaxtyped, Array
from beartype import beartype
from bsplx import design_matrix
from functools import partial

from decoupling._common import dtype_factors, get_design_matrix

class _Decoupling:

    rank: int
    factors: dtype_factors

    @jaxtyped(typechecker=beartype)
    def __init__(self, factors: dtype_factors):
        assert len(factors) >= 3
        assert factors[0].shape[-1] == factors[1].shape[-1] == factors[2].shape[-1]

        if len(factors) == 4: self.W, self.V, self.H, self.R = factors
        else: self.W, self.V, self.H = factors

        self.factors = factors
        self.rank = self.W.shape[1]

    def forward(self, inputs):
        return self.W @ self.internals(self.V.T @ inputs)

    def __call__(self, inputs):
        return self.forward(inputs)

class DecouplingWithPolynomialInternals(_Decoupling):

    degree: int

    def __init__(self, factors, coefs, degree: int):
        super().__init__(factors)

        self.degree = degree
        self.coefs = coefs

        def _single_internal_static(x, coefs):
            return jnp.polyval(jnp.flip(coefs), x)

        self._internals = jax.vmap(_single_internal_static)

    def internals(self, x):
        return self._internals(x, self.coefs)

    def single_internal(self, x, idx: int):
        return jnp.polyval(jnp.flip(self.coefs[idx]), x)

    def _make_polynomial(coefs):
        return partial(jnp.polyval, p=jnp.flip(coefs))

class DecouplingWithSplineInternals(_Decoupling):

    coefs: Array
    knots: Array
    degree: int

    def __init__(self, factors, coefs, knots, degree: int):
        super().__init__(factors)

        self.coefs = jnp.stack(coefs)
        self.knots = jnp.stack(knots)
        self.degree = degree

        def _single_internal_static(x, coefs, knots):
            dm = get_design_matrix(jnp.atleast_1d(x), knots, self.degree)
            return (dm @ coefs).squeeze()

        self._internals = jax.vmap(_single_internal_static)

    def single_internal(self, x, idx: int):
        dm = get_design_matrix(jnp.atleast_1d(x), self.knots[idx], self.degree)
        return (dm @ self.coefs[idx]).squeeze()

    def internals(self, inputs):
        return self._internals(inputs, self.coefs, self.knots)
