import jax, jax.numpy as jnp
from functools import partial
from tqdm import tqdm

from beartype import beartype
from beartype.typing import Optional, Tuple
from jaxtyping import jaxtyped, Array, Float, ArrayLike

from decoupling.utils import cpd_error
from decoupling.result import DecouplingWithPolynomialInternals
from decoupling.algorithm._base import _Base
from decoupling._common import solve_cpd_subproblem, cpd_stopping_criterion
from decoupling._ops import vandermonde_diag, block_diag, lstsq, unfold_kolda, normalize_columns_simple

class BasicDecoupling(_Base):
    '''https://arxiv.org/abs/1410.4060'''

    degree: int
    tol: float

    @jaxtyped(typechecker=beartype)
    def __init__(
        self, 
        rank: int,
        degree: int = 3,
        niters: int = 100,
        ninits: int = 1,
        key: Optional[Array] = None,
        tol: float = 1e-8,
        show_progress: bool = True,
    ):
        super().__init__(rank, niters, ninits, key, show_progress)
        self.degree = degree
        self.tol = tol

    @jaxtyped(typechecker=beartype)
    def run(
        self,
        inputs: Float[ArrayLike, 'N m'],
        outputs: Float[ArrayLike, 'N n'],
        jacobians: Float[ArrayLike, 'n m N'],
    ) -> DecouplingWithPolynomialInternals:

        inputs, outputs, jacobians = self._toarrays(inputs, outputs, jacobians)

        min_error = float('inf')
        best = None

        for key in jax.random.split(self.key, self.ninits):
            factors, weights, errors = self._cpd(jacobians, key)
            error = errors[-1]
            if error < min_error:
                min_error = error
                best = (factors, weights, errors.copy())

        factors, weights, errors = best
        self.errors = jnp.array(errors)

        (W, V, H) = factors
        W = W * weights

        coefs = self._find_coefs(inputs, outputs, W, V)
        return DecouplingWithPolynomialInternals((W, V, H), coefs, self.degree)

    @jaxtyped(typechecker=beartype)
    def _cpd(self, jacobians: Float[Array, 'n m N'], key: Array) -> Tuple[
        Tuple[
            Float[Array, 'n r'], # W
            Float[Array, 'm r'], # V
            Float[Array, 'N r'], # H
        ],
        Float[Array, 'r'], # weights
        list, # errors
    ]:
        '''Canonical Polyadic Decomposition'''

        errors = []
        norm = jnp.linalg.norm(jacobians)

        factors, weights = self._initialize_factors(jacobians, key), jnp.ones(self.rank)
        W, V, H = factors

        solve_W = partial(solve_cpd_subproblem, unfolded=unfold_kolda(jacobians, 0), mode=0)
        solve_V = partial(solve_cpd_subproblem, unfolded=unfold_kolda(jacobians, 1), mode=1)
        solve_H = partial(solve_cpd_subproblem, unfolded=unfold_kolda(jacobians, 2), mode=2)

        bar = tqdm(range(self.niters), type(self).__name__, disable=not self.show_progress)
        for iteration in bar:
            W, _       = normalize_columns_simple(solve_W(W=W, V=V, H=H))
            V, _       = normalize_columns_simple(solve_V(W=W, V=V, H=H))
            H, weights = normalize_columns_simple(solve_H(W=W, V=V, H=H))

            factors = (W, V, H)

            error = cpd_error(jacobians, factors, weights)

            if iteration > 0:
                diff = abs(error - errors[-1])
                bar.set_postfix_str(f'error={error:.4f}, diff={diff:.8f}')

                if cpd_stopping_criterion(diff, self.tol, norm):
                    bar.set_postfix_str(f'(Early stopping after {iteration+1} iterations.)')
                    errors.append(error)
                    break

            else: bar.set_postfix_str(f'error={error:.4f}')

            errors.append(error)

        return factors, weights, errors

    def _find_coefs(
        self,
        X: Float[Array, 'N m'],
        Y: Float[Array, 'N n'],
        W: Float[Array, 'n r'],
        V: Float[Array, 'm r'],
    ) -> Float[Array, 'r d']:

        '''Recover the polynomial internals by fitting a linear system.'''

        N = X.shape[0]
        W_diag = block_diag([W for _ in range(N)])
        Z = jnp.array([V.T @ x for x in X])
        Xk = jnp.concatenate([vandermonde_diag(z, self.degree) for z in Z], axis=0)
        coefs = lstsq(W_diag @ Xk, jnp.concatenate(Y)).T.reshape(self.rank, self.degree+1)
        return coefs
