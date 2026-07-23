import warnings
from tqdm import tqdm
import jax, jax.numpy as jnp
from beartype import beartype
from functools import partial
from beartype.typing import Tuple, NamedTuple
from jaxtyping import jaxtyped, Float, Array

from decoupling.types import *
from decoupling import _ops as ops
from decoupling.utils import cpd_error
from decoupling.result import Decoupling
from decoupling._splines import design_matrix, design_dmatrix

class Information(NamedTuple):
    errors_: Array  # cpd error for each iteration
    lambdas_: Array # smoothing terms for each internal
    rconds_: dict[str, Array] # reciprocal condition numbers for each factor

class Algorithm:

    info: Information

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
        use_smoothing: bool = True,
        lam_nvalues_init: int = 256,
        lam_nvalues: int = 32,
        show_progress: bool = True,
    ):

        assert all(map(lambda x: x > 0, [rank, ninits, niters, splines_degree]))
        assert gamma >= 0.0

        if splines_dof < splines_degree + 1:
            raise ValueError(f'splines_dof ({splines_dof}) must be >= splines_degree + 1 ({splines_degree + 1})')

        self.rank = rank
        self.niters = niters
        self.ninits = ninits 
        self.key = key
        self.gamma = jnp.asarray(gamma)
        self.splines_dof = splines_dof
        self.splines_degree = splines_degree
        self.show_progress = show_progress
        self.use_smoothing = use_smoothing
        self.lam_nvalues_init = lam_nvalues_init
        self.lam_nvalues = lam_nvalues

        self.initial_log_lam_grid = (-6, 3)

    @jaxtyped(typechecker=beartype)
    def run(self, inputs: X_dtype, outputs: Y_dtype, jacobians: J_dtype) -> Decoupling:

        # convert to jax arrays and unfold jacobians
        inputs, outputs, jacobians = self._convert_inputs(inputs, outputs, jacobians)
        unfoldings = self._unfold_jacobians(jacobians)

        result, min_error = None, float('inf')
        for i, key in enumerate(jax.random.split(self.key, self.ninits)):
            factors, coefs_knots, errors, rconds, lambdas = self._run_once(i, key, inputs, outputs, jacobians, unfoldings)

            if (error := min(errors)) < min_error:
                min_error = error
                result = (factors, coefs_knots)
                self.info = Information(jnp.asarray(errors), jnp.asarray(lambdas), {k: jnp.asarray(rconds[k]) for k in rconds.keys()})

        if result is None: raise RuntimeError('all seeds failed')

        factors, (coefs, knots) = result
        return Decoupling(factors, coefs, knots, self.splines_degree)

    @jaxtyped(typechecker=beartype)
    def _run_once(self, i: int, key: Array, inputs: X_dtype, outputs: Y_dtype, jacobians: J_dtype, unfoldings: Tuple) -> Tuple:
        J0, J1, J2 = unfoldings

        errors = []

        (W, V, H, R) = self._initialize_factors(jacobians, key)

        lambdas = []
        log_lams = [None] * self.rank

        rconds = {'W': [], 'V': [], 'H': [], 'R': []}

        bar = tqdm(range(self.niters), desc=f'[Seed {i+1}/{self.ninits}]', disable=not self.show_progress)
        for iteration in bar:
            if self.gamma > 0.0:
                W, rcondW = ops.cmtf_lstsq(ops.khatri_rao(H, V), R, J0, outputs, self.gamma)
            else: W, rcondW = ops.lstsq(ops.khatri_rao(H, V), J0)

            V, rcondV = ops.lstsq(ops.khatri_rao(H, W), J1)
            W, V = ops.normalize_columns_V(W, V)

            H, rcondH = ops.lstsq(ops.khatri_rao(V, W), J2)
            R, rcondR = ops.lstsq(W, outputs.T)

            rconds['W'].append(rcondW)
            rconds['V'].append(rcondV)
            rconds['H'].append(rcondH)
            rconds['R'].append(rcondR)

            min_rcond = min(rcondW, rcondV, rcondH, rcondR)
            if min_rcond < 1e-12: warnings.warn(f'min_rcond={min_rcond:.1e} is lower than 1e-12')

            H, R, (coefs, knots), log_lams = self._projection(H, R, inputs @ V, log_lams)

            error = cpd_error(jacobians, (W, V, H))
            errors.append(error)

            if iteration == 0 or error < best_error:
                best_factors = (W, V, H, R)
                best_coefs_knots = (coefs, knots)
                best_iter = iteration
                best_error = error

            bar.set_postfix_str(f'error={error:.4f}, best={best_error:.4f} ({best_iter}), rcond={min_rcond:.1e}')

            lambdas.append(log_lams)

        return (best_factors, best_coefs_knots, errors, rconds, lambdas)

    @jaxtyped(typechecker=beartype)
    def _convert_inputs(self, inputs: X_dtype, outputs: Y_dtype, jacobians: J_dtype,
    ) -> Tuple[X_dtype, Y_dtype, J_dtype]:
        return (ops.convert_array(inputs), ops.convert_array(outputs), ops.convert_array(jacobians))

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
        R = jax.random.normal(keys[3], shape=(N, self.rank))
        return (W, V, H, R)

    @staticmethod
    @jax.jit
    def _bspline_project(i, coefs, B, dB, H, R):
        H = H.at[:, i].set(dB @ coefs)
        R = R.at[:, i].set(B @ coefs)
        return (H, R)

    def _design_matrices(self, x, knots):
        B = design_matrix(x, knots, self.splines_degree)
        dB = design_dmatrix(x, knots, self.splines_degree)
        return (B, dB)

    @jaxtyped(typechecker=beartype)
    def _projection(self, H: H_dtype, R: R_dtype, Z: Float[Array, 'N r'], prev_log_lams) -> Tuple:

        coefs_out, knots_out = [], []
        new_log_lams = []

        for rank in range(H.shape[1]):
            z, h, r = Z[:, rank], H[:, rank], R[:, rank]

            is_degenerate = (jnp.max(z) - jnp.min(z)) < 1e-6

            if is_degenerate:
                warnings.warn(f'internal {rank} is degenerate (max-min < 1e-6)')
                H = H.at[:, rank].set(jnp.zeros_like(H[:, rank]))
                R = R.at[:, rank].set(jnp.zeros_like(R[:, rank]))
                coefs_out.append(None); knots_out.append(None); new_log_lams.append(None)
                continue

            knots = self._determine_knots(z)

            B, dB = self._design_matrices(z, knots)
            A = jnp.vstack([dB, jnp.sqrt(self.gamma)*B])
            y = jnp.concatenate([h, jnp.sqrt(self.gamma)*r])

            if self.use_smoothing:
                D = ops.second_difference_matrix(B.shape[1])
                log_lam = self._gcv_grid_search(A, y, D, len(z), prev_log_lams[rank])
                new_log_lams.append(log_lam)

                A = jnp.concatenate([A, jnp.sqrt(10**log_lam) * D])
                y = jnp.concatenate([y, jnp.zeros(D.shape[0])])

            coefs, _ = ops.lstsq(A, y)
            H, R = self._bspline_project(rank, coefs, B, dB, H, R)

            # return the coefs for fitting the internals later
            coefs_out.append(coefs)
            knots_out.append(knots)

        return H, R, (coefs_out, knots_out), new_log_lams

    def _gcv_grid_search(self, X, y, D, n, log_lam) -> Array:
        y = jnp.concatenate([y, jnp.zeros(D.shape[0])])
        score = partial(self._gcv_score, X=X, D=D, y=y, n=n)
        best = lambda grid: grid[jnp.argmin(jax.vmap(score)(grid))]
        if log_lam is None: log_lam = best(jnp.linspace(*self.initial_log_lam_grid, self.lam_nvalues_init))
        return best(jnp.linspace(log_lam - 1, log_lam + 1, self.lam_nvalues))

    @staticmethod
    @jax.jit(static_argnames='n')
    def _gcv_score(log_lam: Array, X: Array, D: Array, y: Array, n: int) -> Array:
        lam = 10.0 ** log_lam
        Xa = jnp.concatenate([X, jnp.sqrt(lam) * D])
        Q, R = jnp.linalg.qr(Xa)
        coefs = jax.scipy.linalg.solve_triangular(R, Q.T @ y)
        residuals = y[:2*n] - Xa[:2*n] @ coefs
        rss = jnp.sum(residuals**2)
        df = jnp.sum(Q[:2*n]**2)
        return rss / ((2*n-df)**2)

    def _determine_knots(self, z: Float[Array, 'r']) -> Array:
        if self.use_smoothing: return Algorithm._determine_knots_even(z, self.splines_dof, self.splines_degree)
        return Algorithm._determine_knots_quantiles(z, self.splines_dof, self.splines_degree)

    @staticmethod
    @jax.jit(static_argnames=('dof', 'degree'))
    def _determine_knots_even(u: Float[Array, 'r'], dof: int, degree: int) -> Array:
        internals = dof - degree + 1
        knots = jnp.linspace(jnp.min(u), jnp.max(u), internals)
        begin = jnp.repeat(knots[0], degree)
        end   = jnp.repeat(knots[-1], degree)
        return jnp.concat([begin, knots, end])

    @staticmethod
    @jax.jit(static_argnames=('dof', 'degree'))
    def _determine_knots_quantiles(u: Float[Array, 'r'], dof: int, degree: int) -> Array:
        internals = dof - degree + 1

        qs = jnp.linspace(0, 1, internals)
        knots = jnp.quantile(u, qs)
        knots = jax.vmap(partial(Algorithm._closest, u=u))(knots)

        begin = jnp.repeat(knots[0], degree)
        end = jnp.repeat(knots[-1], degree)
        return jnp.concat([begin, knots, end])

    @staticmethod
    @jax.jit
    def _closest(knot, u):
        def forloop(i, args):
            min_dist, closest_x = args
            x = u[i]
            dist = jnp.abs(x - knot)
            return jax.lax.cond(dist < min_dist, lambda: (dist, x), lambda: (min_dist, closest_x))
        _, closest_point = jax.lax.fori_loop(0, len(u), forloop, (jnp.inf, u[0]))
        return closest_point
