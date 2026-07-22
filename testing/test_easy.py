import jax, jax.numpy as jnp
import unittest

from decoupling.utils import collect_information, function_error
from decoupling.algorithm import Algorithm

def target(x):
    a, b = x
    return jnp.array([
        a**3 + b**2 + a*b,
        b**3 + a**2 + a*b,
    ])

class TestEasy(unittest.TestCase):
    def setUp(self):
        self.key = jax.random.key(0)
        self.rank = 4
        self.ninits = 5
        self.X, self.Y, self.J = collect_information(target, 30, self.key)

    def test_algorithm(self):
        algo = Algorithm(
            rank=self.rank, 
            niters=10, 
            ninits=10, 
            key=self.key, 
            gamma=0.1, 
            splines_dof=30, 
            splines_degree=3, 
            use_smoothing=True,
        )

        decoupling = algo.run(self.X, self.Y, self.J)
        errors = function_error(target, decoupling, self.X, self.key)
        print(errors)
        self.assertTrue(all(jnp.array(errors) < 10.0))
