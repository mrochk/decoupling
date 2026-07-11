import jax, jax.numpy as jnp

from jaxtyping import jaxtyped, Float, Array
from beartype import beartype
from beartype.typing import Tuple, Optional

from decoupling.algorithm._cmtf import _CMTFWithProjection
from decoupling.result import Decoupling
from decoupling import _ops as ops
from decoupling import _common as c

class CMTF_BSpline(_CMTFWithProjection):

    '''https://arxiv.org/abs/2504.03263'''

    dof: int
    degree: int

    def __init__(
        self, 
        rank: int, 
        niters: int = 100, 
        ninits: int = 1, 
        dof: Optional[int] = None,
        degree: int = 3,
        gamma: float = 0.1, 
        key: Optional[Array] = None, 
        show_progress: bool = True,
    ):
        super().__init__(rank, niters, ninits, key, gamma, show_progress)
        self.degree = degree
        self.dof = dof

    def _projection(
        self,
        H: Float[Array, 'N r'],
        R: Float[Array, 'N r'],
        Z: Float[Array, 'N r'],
    ) -> Tuple[Float[Array, 'N r'], Float[Array, 'N r'], Tuple]:

        coefs_out, knots_out = [], []

        for rank in range(H.shape[1]):
            z, h, r = Z[:, rank], H[:, rank], R[:, rank]

            knots = c.determine_knots(z, self.dof, self.degree, 'quantile')

            B = c.get_design_matrix(z, knots, self.degree)
            dB = c.get_design_dmatrix(z, knots, self.degree)

            coefs = ops.cmtf_lstsq(dB, B, h, r, self.gamma)
            H, R = c.bspline_project(rank, coefs, B, dB, H, R)

            g_coefs = jnp.linalg.lstsq(B, R[:, rank])[0]
            coefs_out.append(g_coefs)
            knots_out.append(knots)

        out = (coefs_out, knots_out)

        return H, R, out

    def run(self, inputs, outputs, jacobians):
        if self.dof is None: self.dof = c.default_dof(inputs.shape[0]) 
        factors, (coefs, knots) = self._run(inputs, outputs, jacobians)
        def internals(z): return c.apply_internals(z, coefs, knots, self.degree)
        return Decoupling(factors, internals)
