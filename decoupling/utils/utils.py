import jax, jax.numpy as jnp
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array
from beartype.typing import Callable, Tuple, Optional

from decoupling.types import *
from decoupling import _ops as ops

def find_ninputs(function: Callable):
    assert callable(function)
    for ninputs in range(1, 100_001):
        try: function(jnp.zeros(ninputs)); return ninputs
        except (ValueError, TypeError): pass
        except Exception as e: raise RuntimeError(str(e)) 

def collect_information_from_random(
    function: Callable,
    N: int, 
    key: Array,
    n_inputs: Optional[int] = None, 
    minval: float = 0.0, maxval: float = 1.0,
)-> Tuple[X_dtype, Y_dtype, J_dtype]:
    assert callable(function)
    if n_inputs is None: n_inputs = find_ninputs(function)
    jacobian = jax.jit(jax.vmap(jax.jacobian(function)))
    function = jax.jit(jax.vmap(function))
    inputs = jax.random.uniform(key, shape=(N, n_inputs), minval=minval, maxval=maxval)
    outputs, jacobians = function(inputs), jacobian(inputs)
    return (inputs, outputs, jacobians.transpose((1, 2, 0)))

def collect_information_from_inputs(
    function: Callable,
    inputs: Float[Array, 'N m'],
)-> Tuple[X_dtype, Y_dtype, J_dtype]:
    assert callable(function)
    jacobian = jax.jit(jax.vmap(jax.jacobian(function)))
    function = jax.jit(jax.vmap(function))
    outputs, jacobians = function(inputs), jacobian(inputs)
    return (inputs, outputs, jacobians.transpose((1, 2, 0)))

@jaxtyped(typechecker=beartype)
def cpd_reconstruct(factors: factors_dtype, weights: Optional[Float[Array, 'r']] = None) -> J_dtype:
    rank = factors[0].shape[1]
    if weights is None: weights = jnp.ones(rank)
    return ops.reconstruct(*factors, weights)

@jaxtyped(typechecker=beartype)
def cpd_error(tensor: J_dtype, factors: factors_dtype, weights: Optional[Float[Array, 'r']] = None) -> Float[Array, '']:
    _tensor = cpd_reconstruct(factors, weights)
    return jnp.linalg.norm(tensor - _tensor) / jnp.linalg.norm(tensor)

def function_error(target: Callable, decoupling: Callable, inputs) -> float:
    assert callable(target) and callable(decoupling)
    Y_target = jax.vmap(target)(inputs)
    Y_decoupling = jax.vmap(decoupling)(inputs)
    top = jnp.sqrt(jnp.mean((Y_target - Y_decoupling)**2, axis=0))
    bot = jnp.sqrt(jnp.mean((Y_target - jnp.mean(Y_target, axis=0))**2, axis=0))
    return top / bot * 100
