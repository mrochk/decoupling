import pytest
import jax, jax.numpy as jnp

from decoupling.utils import collect_information, function_error
from decoupling.algorithm import Algorithm

def target(x):
    a, b = x
    return jnp.array([
        a**3 + b**2 + a*b,
        b**3 + a**2 + a*b,
    ])

key = jax.random.key(0)
rank = 4
ninits = 5
X, Y, J = collect_information(target, 30, key)

def test_algorithm():
    algo = Algorithm(
        rank=rank, 
        niters=100, 
        ninits=3, 
        key=key, 
        gamma=0.1, 
        splines_dof=10, 
        splines_degree=3, 
        use_smoothing=True,
    )

    decoupling = algo.run(X, Y, J)
    errors = function_error(target, decoupling, X, key)
    print(errors)
    assert all(jnp.array(errors) < 10.0)
