import jax, jax.numpy as jnp
from tqdm import tqdm
from jaxtyping import jaxtyped, Float, Array, ArrayLike
from beartype.typing import Callable, Tuple, Optional
from beartype import beartype

from decoupling._common import get_random_key, find_number_inputs
from decoupling import _ops as ops

@jaxtyped(typechecker=beartype)
def collect_information(
    function: Callable[[ArrayLike], Array],
    N: int, 
    key: Optional[Array] = None,
    n_inputs: Optional[int] = None, 
    minval: float = 0.0, 
    maxval: float = 1.0,
    X: Optional[Float[Array, 'N m']] = None,
) -> Tuple[Float[Array, 'N m'], Float[Array, 'N n'], Float[Array, 'n m N']]:

    '''
    Generates random inputs X and collects outputs Y
    and stacked jacobian tensor J.
    '''

    assert callable(function)

    if n_inputs is None: n_inputs = find_number_inputs(function)
    if key is None: key = get_random_key()

    jacobian = jax.jit(jax.vmap(jax.jacobian(function)))
    function = jax.jit(jax.vmap(function))

    if X is None: X = jax.random.uniform(key, shape=(N, n_inputs), minval=minval, maxval=maxval)
    Y = function(X)
    J = jacobian(X)

    return (X, Y, J.transpose((1, 2, 0)))

@jaxtyped(typechecker=beartype)
def cpd_reconstruct(
    factors: Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'N r']],
    weights: Optional[Float[Array, 'r']] = None,
) -> Float[Array, 'n m N']:
    rank = factors[0].shape[1]
    if weights is None: weights = jnp.ones(rank)
    return ops.reconstruct(*factors, weights)

@jaxtyped(typechecker=beartype)
def cpd_error(
    tensor: Float[Array, 'n m N'],
    factors: Tuple[Float[Array, 'n r'], Float[Array, 'm r'], Float[Array, 'N r']],
    weights: Optional[Float[Array, 'r']] = None,
) -> Float[Array, '']:
    _tensor = cpd_reconstruct(factors, weights)
    return jnp.linalg.norm(tensor - _tensor) / jnp.linalg.norm(tensor)

def function_error(
    target: Callable,
    decoupling: Callable, 
    X: Optional[ArrayLike] = None,
    key: Optional[Array] = None,
    n_inputs: Optional[int] = None,
    minval: float = 0.0,
    maxval: float = 1.0,
    N: int = 1000,
) -> float:
    assert callable(target) and callable(decoupling)

    if X is None: 
        if key is None: key = get_random_key()
        if n_inputs is None: n_inputs = find_number_inputs(decoupling)
        X = jax.random.uniform(key, shape=(N, n_inputs), minval=minval, maxval=maxval)

    Ytarget  = jax.vmap(target)(X)
    Ylearned = jax.vmap(decoupling)(X)

    top = jnp.sqrt(jnp.mean((Ytarget - Ylearned)**2, axis=0))
    bot = jnp.sqrt(jnp.mean((Ytarget - jnp.mean(Ytarget, axis=0))**2, axis=0))

    return top / bot * 100

from decoupling.result import DecouplingWithSplineInternals

def linearity_r2(x, y, eps=1e-6):
    A = jnp.stack([x, jnp.ones_like(x)], axis=1)
    coef = jnp.linalg.lstsq(A, y)[0]
    y_hat = A @ coef
    ss_res = jnp.sum((y - y_hat)**2)
    ss_tot = jnp.sum((y - jnp.mean(y))**2)
    slope, bias = coef

    if ss_tot < eps * jnp.sum(y**2): # if constant
        return jnp.float32(1.0), slope, bias

    return 1 - ss_res / ss_tot, slope, bias

def find_linear_internals(decoupling: DecouplingWithSplineInternals, X, r2_threshold = 0.98):
    Z = X @ decoupling.V
    points = jax.vmap(decoupling.internals)(Z)

    linear = {}

    for idx, (x, y) in enumerate(zip(Z.T, points.T)):
        r2, a, b = linearity_r2(x, y)
        if r2 >= r2_threshold: linear[idx] = (r2, a, b)

    return linear
