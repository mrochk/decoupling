import jax, jax.numpy as jnp
from beartype import beartype
from jaxtyping import jaxtyped, Float, Array
from beartype.typing import Callable, Optional

from decoupling.result import Decoupling
from decoupling import _ops as ops
from decoupling.types import *
from decoupling.samples import Samples

@jaxtyped(typechecker=beartype)
def collect_samples(function: Callable, inputs: X_dtype)-> Samples:
    '''Collect inputs, outputs and jacobians from `function`.'''
    assert callable(function)
    jacobian = jax.jit(jax.vmap(jax.jacobian(function)))
    function = jax.jit(jax.vmap(function))
    outputs, jacobians = function(inputs), jacobian(inputs)
    return Samples(inputs, outputs, jacobians.transpose((1, 2, 0)))

@jaxtyped(typechecker=beartype)
def collect_samples_from_random(
    key: Array,
    function: Callable, 
    ninputs: int,
    N: int, 
    minval: float = 0.0,
    maxval: float = 1.0,
)-> Samples:
    '''Generate random samples x ~ U(min, max) and collect inputs, outputs and jacobians from `function`.'''
    assert callable(function)
    inputs = jax.random.uniform(key, shape=(N, ninputs), minval=minval, maxval=maxval)
    return collect_samples(function, inputs)

@jaxtyped(typechecker=beartype)
def cpd_reconstruct(factors: factors_dtype, weights: Optional[Float[Array, 'r']] = None) -> J_dtype:
    '''Calls jit-compiled `ops.reconstruct`.'''
    rank = factors[0].shape[1]
    if weights is None: weights = jnp.ones(rank)
    return ops.reconstruct(*factors, weights)

@jaxtyped(typechecker=beartype)
def cpd_error(tensor: J_dtype, factors: factors_dtype, weights: Optional[Float[Array, 'r']] = None) -> Float[Array, '']:
    '''Compute the cpd-error from target tensor, factors and weights.'''
    _tensor = cpd_reconstruct(factors, weights)
    return jnp.linalg.norm(tensor - _tensor) / jnp.linalg.norm(tensor)

@jax.jit
@jaxtyped(typechecker=beartype)
def function_error(Y_true: Y_dtype, Y_pred: Y_dtype) -> Float[Array, 'n']:
    '''Compute the per-output error between targets and predictions.'''
    top = jnp.sqrt(jnp.mean((Y_true - Y_pred)**2, axis=0))
    bot = jnp.sqrt(jnp.mean((Y_true - jnp.mean(Y_true, axis=0))**2, axis=0))
    return top / bot * 100

@jaxtyped(typechecker=beartype)
def function_error_from_callable(target: Callable, decoupling: Decoupling, inputs: X_dtype) -> Float[Array, 'n']:
    '''Compute the per-output error between target function and decoupling using samples `X`.'''
    Y_true = jax.vmap(target)(inputs)
    Y_pred = jax.vmap(decoupling)(inputs)
    return function_error(Y_true, Y_pred)
