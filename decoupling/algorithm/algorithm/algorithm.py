import jax, jax.numpy as jnp
import warnings
from tqdm import tqdm
from jaxtyping import jaxtyped, Float, Array, ArrayLike
from beartype import beartype
from beartype.typing import Tuple, Optional

from decoupling._common import as_float_array, dtype_factors, determine_knots, get_design_matrix, get_design_dmatrix, bspline_project
from decoupling.utils import cpd_error
from decoupling.result import DecouplingWithSplineInternals
from decoupling import _ops as ops

class Algorithm:

    def __init__(
        self,
        rank: int,
        niters: int,
        ninits: int,
        key: Array,
        gamma: float,
        dof: int,
        degree: int,
        show_progress: bool = True,
    ):
        self.rank = rank
        self.niters = niters
        self.ninits = ninits 
        self.key = key
        self.gamma = gamma
        self.dof = dof
        self.degree = degree
        self.show_progress = show_progress
        self.is_cmtf = gamma > 0.0

    def run(self, inputs, outputs, jacobians):
        inputs, outputs, jacobians = self._convert_inputs(inputs, outputs, jacobians)

        J0 = ops.unfold_kolda(jacobians, 0).T
        J1 = ops.unfold_kolda(jacobians, 1).T
        J2 = ops.unfold_kolda(jacobians, 2).T
        unfoldings = (J0, J1, J2)

        min_error = float('inf')

        for key in jax.random.split(self.key, self.ninits):
            factors, coefs_knots, errors = self._run_once(key, inputs, outputs, jacobians, unfoldings)

            if (error := min(errors)) < min_error:
                min_error = error
                result = (factors, coefs_knots)
                self.error = jnp.array(errors)

        factors, (coefs, knots) = result
        return DecouplingWithSplineInternals(factors, coefs, knots, self.degree)

    def _run_once(self, key, inputs, outputs, jacobians, unfoldings):
        J0, J1, J2 = unfoldings

        errors = []

        (W, V, H, R) = self._initialize_factors(jacobians, key)

        self.prev_lams = jnp.linspace(-6, 3, 100)

        bar = tqdm(range(self.niters), type(self).__name__, disable=not self.show_progress)
        for iteration in bar:
            W = ops.cmtf_lstsq(ops.khatri_rao(H, V), R, J0, outputs, self.gamma)
            V = ops.lstsq(ops.khatri_rao(H, W), J1)
            W, V = ops.normalize_columns_V(W, V)

            H = ops.lstsq(ops.khatri_rao(V, W), J2)
            R = ops.lstsq(W, outputs.T)

            H, R, (coefs, knots) = self._projection(H, R, inputs @ V)

            error = cpd_error(jacobians, (W, V, H))
            errors.append(error)

            if iteration == 0 or error < best_error:
                best_factors = (W, V, H, R)
                best_coefs_knots = (coefs, knots)
                best_iter = iteration
                best_error = error

            bar.set_postfix_str(f'error={error:.4f}, best={best_error:.4f} ({best_iter})')

        return (best_factors, best_coefs_knots, errors)

    def _convert_inputs(self, inputs: Float[ArrayLike, 'N m'], outputs: Float[ArrayLike, 'N n'], jacobians: Float[ArrayLike, 'n m N'],
    ) -> Tuple[Float[Array, 'N m'], Float[Array, 'N n'], Float[Array, 'n m N']]:
        return (as_float_array(inputs), as_float_array(outputs), as_float_array(jacobians))

    @jaxtyped(typechecker=beartype)
    def _initialize_factors(self, jacobians: Float[Array, 'n m N'], key: Array) -> dtype_factors:
        n, m, N = jacobians.shape
        keys = jax.random.split(key, num=4)

        W = jax.random.normal(keys[0], shape=(n, self.rank))
        V = jax.random.normal(keys[1], shape=(m, self.rank))
        H = jax.random.normal(keys[2], shape=(N, self.rank))

        if not self.is_cmtf: return (W, V, H)

        R = jax.random.normal(keys[3], shape=(N, self.rank))
        return (W, V, H, R)

    def _projection(self, H: Float[Array, 'N r'], R: Float[Array, 'N r'], Z: Float[Array, 'N r'],
    ) -> Tuple[Float[Array, 'N r'], Float[Array, 'N r']]:

        coefs_out, knots_out = [], []
        new_lams = []

        for rank in range(H.shape[1]):
            z, h, r = Z[:, rank], H[:, rank], R[:, rank]

            is_degenerate = (jnp.max(z) - jnp.min(z)) < 1e-6

            if is_degenerate:
                warnings.warn(f'Internal {rank} is degenerate (max - min < 1e-6).')
                H = H.at[:, rank].set(jnp.zeros_like(H[:, rank]))
                R = R.at[:, rank].set(jnp.zeros_like(R[:, rank]))
                coefs_out.append(None); knots_out.append(None); new_lams.append(None)
                continue

            knots = determine_knots(z, self.dof, self.degree, 'even')
            B = get_design_matrix(z, knots, self.degree)
            dB = get_design_dmatrix(z, knots, self.degree)
            D = ops.second_diff_matrix(B.shape[1])
            A = jnp.vstack([dB, jnp.sqrt(self.gamma)*B])
            y = jnp.concatenate([h, jnp.sqrt(self.gamma)*r])
            ll = self._gcv_grid_search(A, y, D, len(z), self.prev_lams[rank], 100, 100)

            new_lams.append(ll)
            lam = 10**ll

            A = jnp.concatenate([A, jnp.sqrt(lam) * D])
            y = jnp.concatenate([y, jnp.zeros(D.shape[0])])
            coefs = ops.lstsq(A, y).T
            H, R = bspline_project(rank, coefs, B, dB, H, R)

            # return the coefs for fitting the internals later
            coefs_out.append(ops.lstsq(B, R[:, rank]).T)
            knots_out.append(knots)

        self.prev_lams = new_lams
        return H, R, (coefs_out, knots_out)

    def _gcv_grid_search(
        self,
        X: Array, 
        y: Array, 
        D: Array, 
        n: int, 
        _ll: Optional[float],
        nvalues_init: int,
        nvalues: int,
    ) -> Array:

        y = jnp.concatenate([y, jnp.zeros(D.shape[0])])

        if _ll is None: # if first iteration
            lls_init = jnp.linspace(-6, 3, nvalues_init)
            scores = jax.vmap(lambda ll: self._gcv_score(ll, X, D, y, n))(lls_init)
            _ll = lls_init[jnp.argmin(scores)]

        lls = jnp.linspace(_ll-1, _ll+1, nvalues)
        scores = jax.vmap(lambda ll: self._gcv_score(ll, X, D, y, n))(lls)
        return lls[jnp.argmin(scores)]

    @staticmethod
    @jax.jit(static_argnames='n')
    def _gcv_score(ll: Array, X: Array, D: Array, y: Array, n: int) -> Array:
        lam = 10.0 ** ll
        X = jnp.concatenate([X, jnp.sqrt(lam)*D])
        coefs = ops.lstsq(X, y).T
        residuals = y[:2*n] - X[:2*n] @ coefs
        rss = jnp.sum(residuals ** 2)
        Q = jnp.linalg.qr(X)[0]
        df = jnp.trace(Q[:2*n] @ Q[:2*n].T)
        return (n * rss) / (n - df)**2
