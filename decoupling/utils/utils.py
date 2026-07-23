import jax, jax.numpy as jnp
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array
from beartype.typing import Callable, Tuple, Optional

from decoupling.result import Decoupling
from decoupling.types import *
from decoupling import _ops as ops

@jaxtyped(typechecker=beartype)
def collect_information_from_random(
    function: Callable,
    N: int, 
    ninputs: int, 
    key: Array,
    minval: float = 0.0, maxval: float = 1.0,
)-> Tuple[X_dtype, Y_dtype, J_dtype]:
    ''' generate random samples and return inputs, outputs and jacobians '''
    assert callable(function)
    if ninputs is None: ninputs = find_ninputs(function)
    jacobian = jax.jit(jax.vmap(jax.jacobian(function)))
    function = jax.jit(jax.vmap(function))
    inputs = jax.random.uniform(key, shape=(N, ninputs), minval=minval, maxval=maxval)
    outputs, jacobians = function(inputs), jacobian(inputs)
    return (inputs, outputs, jacobians.transpose((1, 2, 0)))

@jaxtyped(typechecker=beartype)
def collect_information_from_inputs(function: Callable, inputs: X_dtype)-> Tuple[X_dtype, Y_dtype, J_dtype]:
    ''' return inputs, outputs and jacobians '''
    assert callable(function)
    jacobian = jax.jit(jax.vmap(jax.jacobian(function)))
    function = jax.jit(jax.vmap(function))
    outputs, jacobians = function(inputs), jacobian(inputs)
    return (inputs, outputs, jacobians.transpose((1, 2, 0)))

@jaxtyped(typechecker=beartype)
def cpd_reconstruct(factors: factors_dtype, weights: Optional[Float[Array, 'r']] = None) -> J_dtype:
    ''' calls jit-compiled ops.reconstruct '''
    rank = factors[0].shape[1]
    if weights is None: weights = jnp.ones(rank)
    return ops.reconstruct(*factors, weights)

@jaxtyped(typechecker=beartype)
def cpd_error(tensor: J_dtype, factors: factors_dtype, weights: Optional[Float[Array, 'r']] = None) -> Float[Array, '']:
    ''' compute the cpd error from target tensor, factors and weights '''
    _tensor = cpd_reconstruct(factors, weights)
    return jnp.linalg.norm(tensor - _tensor) / jnp.linalg.norm(tensor)

@jaxtyped(typechecker=beartype)
def function_error(target: Callable, decoupling: Decoupling, inputs: X_dtype) -> Float[Array, 'n']:
    ''' compute the per-output error between target function and decoupling '''
    Y_target = jax.vmap(target)(inputs)
    Y_decoupling = jax.vmap(decoupling)(inputs)
    top = jnp.sqrt(jnp.mean((Y_target - Y_decoupling)**2, axis=0))
    bot = jnp.sqrt(jnp.mean((Y_target - jnp.mean(Y_target, axis=0))**2, axis=0))
    return top / bot * 100
