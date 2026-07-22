import jax, jax.numpy as jnp

from decoupling import Algorithm
from decoupling.scaler import JacobianScaler
from decoupling.utils import collect_information_from_random, function_error

def target(x):
    a, b = x
    return jnp.array([
        1000*(a**3 + b**2 + a*b),
        b**3 + a**2 + a*b,
    ])

N = 100

key = jax.random.key(0)
rank, ninits = 4, 5
X, Y, J = collect_information_from_random(target, N, key)
scaler = JacobianScaler(J)
Js, Ys = scaler.scale(J, Y)

def test_algorithm():

    algorithm = Algorithm(
        rank=rank, 
        niters=100, 
        splines_dof=int(jnp.sqrt(N)), 
        key=key, 
        ninits=3, 
    )

    decoupling = algorithm.run(X, Ys, Js)
    decoupling = scaler.unscale(decoupling)

    errors = function_error(target, decoupling, X)
    assert all(jnp.array(errors) < 10.0)
    print(errors)

    print(jnp.argmin(algorithm.info.errors_))
