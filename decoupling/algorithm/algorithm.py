import warnings
from tqdm import tqdm
import jax, jax.numpy as jnp
from beartype import beartype
from functools import partial
from bsplx import repeat_knots
from beartype.typing import Tuple, NamedTuple, Optional
from jaxtyping import jaxtyped, Float, Array, ArrayLike

from decoupling.types import *
from decoupling import _ops as ops
from decoupling.utils import cpd_error
from decoupling.result import Decoupling
from decoupling._splines import design_matrices

class Algorithm:
    '''Tensor decoupling algorithm class.'''

    class Information(NamedTuple):
        '''Named tuple containing information about the best run.'''
        errors: Array # cpd error at each iteration
        lambdas: Optional[Array] # smoothing terms for each internal
        rconds: dict[str, Array] # reciprocal condition numbers for each factor

    info: Information

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        rank: int,
        key: Array,
        niters: int = 20,
        ninits: int = 1,
        gamma: float = 0.1,
        splines_dof: Optional[int] = None,
        splines_degree: int = 3,
        knots: str = 'quantile',
        use_smoothing: bool = True,
        smoothing_grid: ArrayLike = jnp.linspace(-3, 6, num=2048),
        show_progress: bool = True,
    ):
        '''
        Args:
            rank (int): rank of the decomposition (number of internals) 
            niters (int): number of iterations 
            splines_dof (int): splines (internals) degrees of freedom 
            key (Array): jax random seed 
            ninits (int): number of runs to try from different seeds derived from key 
            gamma (float): weight of zeroth-order information in cmtf objective 
            splines_degree (int): splines degree (default=3) 
            knots (str): the knots placement to use ['quantile' or 'even']
            use_smoothing (bool): whether to use P-splines or B-splines 
            smoothing_grid (ArrayLike): the grid of lambda values to search for, in log space 
            show_progress (bool): whether to show the progress bar 
        '''

        assert all(map(lambda x: x > 0, [rank, ninits, niters, splines_degree]))
        assert gamma >= 0.0

        if splines_dof is not None and splines_dof < splines_degree + 1:
            raise ValueError(f'dof ({splines_dof}) must be >= degree + 1 ({splines_degree + 1})')

        assert knots in {'even', 'quantile'}

        self.rank = rank
        self.niters = niters
        self.ninits = ninits 
        self.key = key
        self.gamma = jnp.asarray(gamma)
        self.splines_dof = splines_dof
        self.splines_degree = splines_degree
        self.knots = knots
        self.use_smoothing = use_smoothing
        self.smoothing_grid = ops.convert_array(smoothing_grid)
        self.show_progress = show_progress

    @jaxtyped(typechecker=beartype)
    def run(self, inputs: X_dtype, outputs: Y_dtype, jacobians: J_dtype) -> Decoupling:
        '''Compute the decoupled representation of target given inputs, outputs and Jacobians.'''

        if self.splines_dof is None:
            self.splines_dof = max(self.splines_degree+1, int(jnp.sqrt(2*inputs.shape[0])))
            warnings.warn(f'splines_dof not provided, setting it to {self.splines_dof}')

        # convert to jax arrays and unfold jacobians
        inputs, outputs, jacobians = self._convert_inputs(inputs, outputs, jacobians)
        unfoldings = self._unfold_jacobians(jacobians)

        result, min_error = None, float('inf')
        for i, key in enumerate(jax.random.split(self.key, self.ninits)):
            factors, coefs_knots, info = self._run_once(i, key, inputs, outputs, jacobians, unfoldings)

            if (error := min(info.errors)) >= min_error or jnp.isnan(error): continue
            if any(map(lambda x: x is None, coefs_knots[0])): continue

            min_error = error
            result = (factors, coefs_knots)
            self.info = info

        if result is None: raise RuntimeError('all seeds failed')

        factors, (coefs, knots) = result
        return Decoupling(factors, coefs, knots, self.splines_degree)

    @jaxtyped(typechecker=beartype)
    def _run_once(
        self, 
        seed: int, 
        key: Array, 
        inputs: X_dtype, 
        outputs: Y_dtype, 
        jacobians: J_dtype, 
        unfoldings: Tuple[Array, Array, Array],
    ) -> Tuple[factors_dtype, Tuple, Information]:

        J0, J1, J2 = unfoldings

        errors = []

        (W, V, H, R) = self._initialize_factors(jacobians, key)

        lambdas = []
        log_lams = [None] * self.rank

        rconds = {'W': [], 'V': [], 'H': [], 'R': []}

        bar = tqdm(range(self.niters), desc=f'[Seed {seed+1}/{self.ninits}]', disable=not self.show_progress)
        for iteration in bar:
            if self.gamma > 0.0: W, rcondW = ops.cmtf_lstsq(ops.khatri_rao(H, V), R, J0, outputs, self.gamma)
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

            H, R, (coefs, knots), log_lams = self._projection(H, R, inputs@V)

            error = cpd_error(jacobians, (W, V, H))
            errors.append(error)

            if iteration == 0 or error < best_error:
                factors = (W, V, H, R)
                coefs_knots = (coefs, knots)
                best_iter = iteration
                best_error = error

            bar.set_postfix_str(f'error={error:.4f}, best={best_error:.4f} ({best_iter}), rcond={min_rcond:.1e}')

            lambdas.append([jnp.nan if l is None else l for l in log_lams])

        errors = jnp.asarray(errors)
        if self.use_smoothing: lambdas = jnp.asarray(lambdas)
        else: lambdas = None
        rconds = {k: jnp.asarray(rconds[k]) for k in rconds.keys()}
        info = Algorithm.Information(errors, lambdas, rconds)

        return (factors, coefs_knots, info)

    @jaxtyped(typechecker=beartype)
    def _convert_inputs(self, inputs: X_dtype, outputs: Y_dtype, jacobians: J_dtype,
    ) -> Tuple[X_dtype, Y_dtype, J_dtype]:
        return (ops.convert_array(inputs), ops.convert_array(outputs), ops.convert_array(jacobians))

    @staticmethod
    @jaxtyped(typechecker=beartype)
    def _unfold_jacobians(jacobians: J_dtype) -> Tuple[Array, Array, Array]:
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
    @jax.jit(static_argnames='rank')
    @jaxtyped(typechecker=beartype)
    def _project(rank: int, coefs: Array, B: Array, dB: Array, H: H_dtype, R: R_dtype) -> Tuple[H_dtype, R_dtype]:
        H = H.at[:, rank].set(dB @ coefs)
        R = R.at[:, rank].set(B @ coefs)
        return (H, R)

    @jaxtyped(typechecker=beartype)
    def _projection(self, H: H_dtype, R: R_dtype, Z: Float[Array, 'N r']) -> Tuple:

        coefs_out, knots_out = [], []
        lambdas = []

        for rank in range(H.shape[1]):
            z, h, r = Z[:, rank], H[:, rank], R[:, rank]

            is_degenerate = (jnp.max(z) - jnp.min(z)) < 1e-6

            if is_degenerate:
                warnings.warn(f'internal {rank} is degenerate (max-min < 1e-6)')
                H = H.at[:, rank].set(jnp.zeros_like(H[:, rank]))
                R = R.at[:, rank].set(jnp.zeros_like(R[:, rank]))
                coefs_out.append(None); knots_out.append(None)
                if self.use_smoothing: lambdas.append(None)
                continue

            knots = self._determine_knots(z)

            B, dB = design_matrices(z, knots, self.splines_degree)

            A = jnp.vstack([dB, jnp.sqrt(self.gamma) * B])
            y = jnp.concatenate([h, jnp.sqrt(self.gamma) * r])

            if self.use_smoothing:
                D = ops.second_difference_matrix(B.shape[1])
                n_eff = (A.shape[0] / 2) * (1.0 + self.gamma)
                (ll, coefs, scores) = Algorithm.gcv_path(A, y, D, self.smoothing_grid, n_eff)
                lambdas.append(ll)
            else:
                coefs = ops.lstsq(A, y)[0]

            H, R = self._project(rank, coefs, B, dB, H, R)

            # return the coefs for fitting the internals later
            coefs_out.append(coefs)
            knots_out.append(knots)

        return H, R, (coefs_out, knots_out), lambdas

    @staticmethod
    @jax.jit
    def gcv_path(A, y, D, grid, N):

        n, p = A.shape

        M = jnp.concatenate([A, D], axis=0)
        Qm, Rm = jnp.linalg.qr(M)
        Um, s, Vmt = jnp.linalg.svd(Rm)

        tol = jnp.finfo(A.dtype).eps * max(M.shape) * s[0]
        keep = s > tol
        inv_s = jnp.where(keep, 1.0 / jnp.where(keep, s, 1.0), 0.0)

        X = Qm[:n] @ jnp.where(keep[None, :], Um, 0.0)  # A @ pinv factor, sv <= 1

        Qx, Rx = jnp.linalg.qr(X)
        Ux, c, Wt = jnp.linalg.svd(Rx)

        f = Ux.T @ (Qx.T @ y)
        r0 = y - Qx @ (Ux @ f)
        rss0 = r0 @ r0

        c2 = jnp.clip(c * c, 0.0, 1.0)
        d2 = 1.0 - c2

        lam = jnp.power(10.0, grid)[:, None]
        denom = c2 + lam * d2
        rss = rss0 + jnp.sum((f * (lam * d2) / denom) ** 2, axis=1)
        dof = jnp.sum(c2 / denom, axis=1)

        scores = (rss / N) / (1.0 - dof / N) ** 2
        scores = jnp.where(jnp.isfinite(scores) & (dof < N), scores, jnp.inf)

        best = jnp.argmin(scores)
        ll = grid[best]

        a = c * f / denom[best]
        coefs = (Vmt.T * inv_s) @ (Wt.T @ a)

        return ll, coefs, scores

    def _determine_knots(self, z: Float[Array, 'r']) -> Array:
        match self.knots:
            case 'even': return Algorithm._determine_knots_even(z, self.splines_dof, self.splines_degree)
            case 'quantile': return Algorithm._determine_knots_quantiles(z, self.splines_dof, self.splines_degree)
            case _: raise ValueError()

    @staticmethod
    @jax.jit(static_argnames=('dof', 'degree'))
    def _determine_knots_even(u: Float[Array, 'r'], dof: int, degree: int) -> Array:
        internals = dof - degree + 1
        knots = jnp.linspace(jnp.min(u), jnp.max(u), internals)
        return repeat_knots(knots, degree)

    @staticmethod
    @partial(jax.jit, static_argnames=('dof', 'degree'))
    def _determine_knots_quantiles(u, dof: int, degree: int, alpha: float = 0.8) -> Array:
        internals = dof - degree + 1
        qs = jnp.linspace(0, 1, internals)

        knots_q = jnp.quantile(u, qs)
        knots_u = jnp.linspace(jnp.min(u), jnp.max(u), internals)

        knots = alpha * knots_q + (1.0 - alpha) * knots_u
        return repeat_knots(knots, degree)
