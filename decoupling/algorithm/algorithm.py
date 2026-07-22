import warnings
from tqdm import tqdm
import jax, jax.numpy as jnp
from beartype import beartype
from beartype.typing import Tuple, Optional
from jaxtyping import jaxtyped, Float, Array

from decoupling.types import *
from decoupling import _ops as ops
from decoupling.utils import cpd_error
from decoupling.result import DecouplingWithSplineInternals
from decoupling._common import (
    as_float_array, 
    determine_knots, 
    get_design_matrix, 
    get_design_dmatrix, 
)

class Algorithm:

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        rank: int,
        niters: int,
        splines_dof: int,
        key: Array,
        ninits: int = 1,
        gamma: float = 0.1,
        splines_degree: int = 3,
        use_smoothing: bool = False,
        show_progress: bool = True,
    ):
        self.rank = rank
        self.niters = niters
        self.ninits = ninits 
        self.key = key
        self.gamma = gamma
        self.splines_dof = splines_dof
        self.splines_degree = splines_degree
        self.show_progress = show_progress
        self.use_smoothing = use_smoothing
        self.is_cmtf = gamma > 0.0

    @jaxtyped(typechecker=beartype)
    def run(self, inputs: X_dtype, outputs: Y_dtype, jacobians: J_dtype) -> DecouplingWithSplineInternals:
        inputs, outputs, jacobians = self._convert_inputs(inputs, outputs, jacobians)
        
        unfoldings = self._unfold_jacobians(jacobians)

        min_error = float('inf')

        for i, key in enumerate(jax.random.split(self.key, self.ninits)):
            factors, coefs_knots, errors = self._run_once(i, key, inputs, outputs, jacobians, unfoldings)

            if (error := min(errors)) < min_error:
                min_error = error
                result = (factors, coefs_knots)
                self.error = jnp.array(errors)

        factors, (coefs, knots) = result
        return DecouplingWithSplineInternals(factors, coefs, knots, self.splines_degree)

    @jaxtyped(typechecker=beartype)
    def _run_once(self, i: int, key: Array, inputs: X_dtype, outputs: Y_dtype, jacobians: J_dtype, unfoldings: Tuple) -> Tuple:
        J0, J1, J2 = unfoldings

        errors = []

        (W, V, H, R) = self._initialize_factors(jacobians, key)

        self.prev_lams = jnp.linspace(-6, 3, 100)

        bar = tqdm(range(self.niters), desc=f'[Seed {i+1}/{self.ninits}]', disable=not self.show_progress)
        for iteration in bar:
            W = ops.cmtf_lstsq(ops.khatri_rao(H, V), R, J0, outputs, self.gamma)
            V, rcond = ops.lstsq(ops.khatri_rao(H, W), J1)
            print(rcond)
            W, V = ops.normalize_columns_V(W, V)

            H, rcond = ops.lstsq(ops.khatri_rao(V, W), J2)
            print(rcond)

            R, rcond = ops.lstsq(W, outputs.T)
            print(rcond)

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

    @jaxtyped(typechecker=beartype)
    def _convert_inputs(self, inputs: X_dtype, outputs: Y_dtype, jacobians: J_dtype,
    ) -> Tuple[X_dtype, Y_dtype, J_dtype]:
        return (as_float_array(inputs), as_float_array(outputs), as_float_array(jacobians))

    @staticmethod
    def _unfold_jacobians(jacobians):
        J0 = ops.unfold_kolda(jacobians, 0).T
        J1 = ops.unfold_kolda(jacobians, 1).T
        J2 = ops.unfold_kolda(jacobians, 2).T
        return (J0, J1, J2)

    @jaxtyped(typechecker=beartype)
    def _initialize_factors(self, jacobians: J_dtype, key: Array) -> factors_dtype:
        n, m, N = jacobians.shape
        keys = jax.random.split(key, num=4)

        W = jax.random.normal(keys[0], shape=(n, self.rank))
        V = jax.random.normal(keys[1], shape=(m, self.rank))
        H = jax.random.normal(keys[2], shape=(N, self.rank))

        if not self.is_cmtf: return (W, V, H)

        R = jax.random.normal(keys[3], shape=(N, self.rank))
        return (W, V, H, R)

    @staticmethod
    @jax.jit
    def bspline_project(i, coefs, B, dB, H, R):
        H = H.at[:, i].set(dB @ coefs)
        R = R.at[:, i].set(B @ coefs)
        return (H, R)

    @jaxtyped(typechecker=beartype)
    def _projection(self, H: H_dtype, R: R_dtype, Z: Float[Array, 'N r']) -> Tuple[H_dtype, R_dtype, Tuple]:

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

            knots = determine_knots(z, self.splines_dof, self.splines_degree, 'even' if self.use_smoothing else 'quantile')
            B = get_design_matrix(z, knots, self.splines_degree)
            dB = get_design_dmatrix(z, knots, self.splines_degree)
            A = jnp.vstack([dB, jnp.sqrt(self.gamma)*B])
            y = jnp.concatenate([h, jnp.sqrt(self.gamma)*r])

            if self.use_smoothing:
                D = ops.second_diff_matrix(B.shape[1])
                ll = self._gcv_grid_search(A, y, D, len(z), self.prev_lams[rank], 100, 100)

                new_lams.append(ll)
                lam = 10**ll

                A = jnp.concatenate([A, jnp.sqrt(lam) * D])
                y = jnp.concatenate([y, jnp.zeros(D.shape[0])])

            coefs = ops.lstsq(A, y)[0].T
            H, R = self.bspline_project(rank, coefs, B, dB, H, R)

            # return the coefs for fitting the internals later
            coefs_out.append(ops.lstsq(B, R[:, rank])[0].T)
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
        coefs = ops.lstsq(X, y)[0].T
        residuals = y[:2*n] - X[:2*n] @ coefs
        rss = jnp.sum(residuals ** 2)
        Q = jnp.linalg.qr(X)[0]
        df = jnp.trace(Q[:2*n] @ Q[:2*n].T)
        return (n * rss) / (n - df)**2
