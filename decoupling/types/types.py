from jaxtyping import Float, Array, ArrayLike
from beartype.typing import Union, Tuple

X_dtype = Float[ArrayLike, 'N m']
Y_dtype = Float[ArrayLike, 'N n']
J_dtype = Float[ArrayLike, 'n m N']

W_dtype = Float[Array, 'n r']
V_dtype = Float[Array, 'm r']
H_dtype = Float[Array, 'N r']
R_dtype = Float[Array, 'N r']

factors_dtype = Union[
    Tuple[W_dtype, V_dtype, H_dtype],
    Tuple[W_dtype, V_dtype, H_dtype, R_dtype],
]
