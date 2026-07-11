import jax, jax.numpy as jnp
from jaxtyping import jaxtyped, Float, Array
from beartype.typing import Tuple, Optional
from beartype import beartype
import warnings

from decoupling._common import *
from decoupling import _ops as ops 
from decoupling.result import Decoupling
from decoupling.algorithm._cmtf import _CMTFWithProjection

class CMTF_PSpline(_CMTFWithProjection):

    dof: int
    degree: int
    lam_nvalues: int
    lam_nvalues_init: int
    prev_lams: list

    def __init__(
        self, 
        rank: int, 
        niters: int = 100, 
        ninits: int = 1, 
        dof: Optional[int] = None,
        degree: int = 3,
        gamma: float = 0.1, 
        lam_nvalues: int = 100,
        lam_nvalues_init: int = 1000,
        key: Optional[Array] = None, 
        show_progress: bool = True,
    ):
        super().__init__(rank, niters, ninits, key, gamma, show_progress)
        self.dof = dof
        self.degree = degree
        self.lam_nvalues = lam_nvalues
        self.lam_nvalues_init = lam_nvalues_init
        self.prev_lams = [None for _ in range(rank)]

    def run(self, inputs, outputs, jacobians):
        if self.dof is None: self.dof = default_dof(inputs.shape[0]) 
        factors, (coefs, knots) = self._run(inputs, outputs, jacobians)
        def internals(z): return apply_internals(z, coefs, knots, self.degree)
        return Decoupling(factors, internals)

    def _projection(
        self,
        H: Float[Array, 'N r'],
        R: Float[Array, 'N r'],
        Z: Float[Array, 'N r'],
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
            ll = self._gcv_grid_search(A, y, D, len(z), self.prev_lams[rank], self.lam_nvalues_init, self.lam_nvalues)

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
