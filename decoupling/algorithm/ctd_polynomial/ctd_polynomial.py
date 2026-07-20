import jax, jax.numpy as jnp
from numpy.polynomial import Polynomial
from functools import partial
from tqdm import tqdm

from jaxtyping import jaxtyped, Array, Float, ArrayLike
from beartype import beartype 
from beartype.typing import Tuple, Optional 

from decoupling.utils import get_random_key, cpd_error
from decoupling import _ops as ops
from decoupling._common import solve_cpd_subproblem
from decoupling.result import DecouplingWithPolynomialInternals
from decoupling.algorithm._base import _Base

class CTD_Polynomial(_Base):

    '''https://hdl.handle.net/20.500.14017/4821c138-dddb-481d-9ea8-8bc6fd939b90'''

    degree: int

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        rank: int,
        degree: int = 3,
        niters: int = 100,
        ninits: int = 1,
        key: Optional[Array] = None,
        show_progress: bool = True,
    ):
        super().__init__(rank, niters, ninits, key, show_progress)
        self.degree = degree

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
            factors, dcoefs, errors = self._cpd_polynomial_constraint(inputs, jacobians, key)
            error = errors[-1]
            if error < min_error:
                min_error = error
                best = (factors, dcoefs, errors.copy())

        factors, dcoefs, errors = best
        self.errors = jnp.array(errors)

        (W, V, _) = factors

        coefs = self._integrate(dcoefs, inputs @ V, outputs, W)
        return DecouplingWithPolynomialInternals(factors, coefs, self.degree)

    @jaxtyped(typechecker=beartype)
    def _cpd_polynomial_constraint(
        self,
        X: Float[Array, 'N m'],
        jacobians: Float[Array, 'n m N'],
        key: Array,
    ) -> Tuple[
        Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'N r']],
        Float[Array, 'd r'], list,
    ]:
        if key is None: key = get_random_key()

        errors = []

        factors, weights = self._initialize_factors(jacobians, key), jnp.ones(self.rank)
        W, V, H = factors

        J0 = ops.unfold_kolda(jacobians, 0)
        J1 = ops.unfold_kolda(jacobians, 1)
        J2 = ops.unfold_kolda(jacobians, 2)

        solve_W = partial(solve_cpd_subproblem, unfolded=J0, mode=0)
        solve_V = partial(solve_cpd_subproblem, unfolded=J1, mode=1)

        min_error = float('inf')
        best = None

        bar = tqdm(range(self.niters), type(self).__name__, disable=not self.show_progress)
        for iteration in bar:
            W = solve_W(W=W, V=V, H=H)
            V = solve_V(W=W, V=V, H=H)
            W, V = ops.normalize_columns_V(W, V)

            H, dcoefs = self._update_H_with_polynomial_constraint(J2, X, V, W)

            factors = W, V, H
            error = cpd_error(jacobians, factors, weights)

            if error < min_error:
                min_error = error
                best = ((W.copy(), V.copy(), H.copy()), dcoefs.copy())
                best_iter = iteration

            if iteration > 0:
                diff = abs(error - errors[-1])
                bar.set_postfix_str(f'error={error:.4f}, best={min_error:.4f} ({best_iter})')

            else: bar.set_postfix_str(f'error={error:.4f}, best={best_iter}')
            errors.append(error)

        return best[0], best[1], errors

    def _update_H_with_polynomial_constraint(
        self,
        unfolded: Array,
        X: Float[Array, 'N m'],
        V: Float[Array, 'm r'],
        W: Float[Array, 'n r'],
    ) -> Tuple[Float[Array, 'N r'], Float[Array, 'd r']]:
        Z = X @ V # inputs to g

        vand_matrices = []
        for r in range(self.rank):
            vand = ops.vandermonde_matrix(Z[:, r], self.degree)
            vand_matrices.append(vand)

        vand_diag = ops.block_diag(vand_matrices)

        KR = ops.khatri_rao(V, W)
        K = jnp.kron(KR, jnp.eye(X.shape[0]))
        Z = K @ vand_diag

        Zinv = jnp.linalg.pinv(Z)
        dcoefs = Zinv @ ops.reshape(unfolded, -1)
        dcoefs = ops.reshape(dcoefs, (self.degree+1, self.rank))

        H = jnp.column_stack([Xi @ ci for Xi, ci in zip(vand_matrices, dcoefs.T)])
        return H, dcoefs

    def _integrate(self, dcoefs, Z, Y, W):
        assert dcoefs.shape[1] == W.shape[1]

        degree, rank = dcoefs.shape
        coefs = jnp.zeros((degree + 1, rank))

        # find non-constant coefficients from integration:
        # c_{i+1} = c'_i / (i+1)
        for d in range(degree):
            coefs = coefs.at[d + 1, :].set(dcoefs[d, :] / (d + 1))

        # estimate constants from observations
        gs_no_czero = [Polynomial(coefs[:, j]) for j in range(rank)]

        N = Z.shape[0]
        n = Y.shape[1]

        columns_W = jnp.zeros((N * n, rank))
        residuals = jnp.zeros(N * n)

        for i in range(N):
            # evaluate g(z) without constants
            g_vals = jnp.array([gs_no_czero[j](Z[i, j]) for j in range(rank)])
            y_pred = W @ g_vals

            for j in range(n):  # for each output
                row_idx = i * n + j  # get the correct index for output j of sample i

                # we are looking for constants that minimize the residual
                # between the y and the values we get when using the polynomials
                # without constant terms
                residuals = residuals.at[row_idx].set(Y[i, j] - y_pred[j])

                # design matrix of our least-squares problem
                # it contains the corresponding column of W for each output
                # (the one that multiplies the c_0 we want to solve for, for the corresponding output)
                columns_W = columns_W.at[row_idx, :].set(W[j, :])

        # solve for constants c_{i, 0}
        c_zeros = jnp.linalg.lstsq(columns_W, residuals, rcond=None)[0]
        coefs = coefs.at[0, :].set(c_zeros)

        return coefs.T
