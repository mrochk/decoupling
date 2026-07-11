import jax, jax.numpy as jnp
from tqdm import tqdm

from beartype import beartype
from beartype.typing import Tuple
from jaxtyping import jaxtyped, Float, ArrayLike

from decoupling import _ops as ops
from decoupling.utils import cpd_error
from decoupling.algorithm._base import _Base
from decoupling._common import dtype_factors

class _CMTFWithProjection(_Base):

    gamma: float

    def __init__(self, rank, niters, ninits, key, gamma, show_progress):
        super().__init__(rank, niters, ninits, key, show_progress)
        self.gamma = gamma

    def _projection(self, H, R, Z):
        pass # return new_H, new_R, out_proj

    @jaxtyped(typechecker=beartype)
    def _run(
        self,
        inputs: Float[ArrayLike, 'N m'],
        outputs: Float[ArrayLike, 'N n'],
        jacobians: Float[ArrayLike, 'n m N'],
    ) -> Tuple[dtype_factors, Tuple]:
        inputs, outputs, jacobians = self._toarrays(inputs, outputs, jacobians)

        best_errors = []

        J0 = ops.unfold_kolda(jacobians, 0).T
        J1 = ops.unfold_kolda(jacobians, 1).T
        J2 = ops.unfold_kolda(jacobians, 2).T

        min_error = float('inf')
        result = None

        for key in jax.random.split(self.key, self.ninits):

            errors = []

            W, V, H, R = self._initialize_factors(jacobians, key, True)

            bar = tqdm(range(self.niters), type(self).__name__, disable=not self.show_progress)

            out_proj = None

            for iteration in bar:
                W = ops.cmtf_lstsq(ops.khatri_rao(H, V), R, J0, outputs, self.gamma)
                V = ops.lstsq(ops.khatri_rao(H, W), J1)
                W, V = ops.normalize_columns_V(W, V)

                H = ops.lstsq(ops.khatri_rao(V, W), J2)
                R = ops.lstsq(W, outputs.T)

                H, R, out_proj = self._projection(H, R, inputs @ V)

                error = cpd_error(jacobians, (W, V, H))
                errors.append(error)

                if iteration == 0 or error < best_error:
                    best = (W, V, H, R)
                    best_iter = iteration
                    best_error = error
                    best_out_proj = out_proj

                bar.set_postfix_str(f'error={error:.4f}, best={best_error:.4f} ({best_iter})')
                best_errors.append(best_error)

            if best_error < min_error:
                min_error = best_error
                result = (best, best_out_proj)
                self.errors = jnp.array(errors)

        return result
