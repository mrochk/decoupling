import jax, jax.numpy as jnp
import unittest

from decoupling.utils import collect_information, function_error, find_linear_internals
from decoupling.algorithm import BasicDecoupling, CTD_Polynomial, CMTF_BSpline, CMTF_PSpline
from decoupling.algorithm.algorithm import Algorithm
from decoupling.result import DecouplingWithSplineAndLinearInternals

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

    #def test_basic(self):
        #algo = BasicDecoupling(rank=self.rank, key=self.key)
        #decoupling = algo.run(self.X, self.Y, self.J)
        #errors = function_error(target, decoupling, self.X, self.key)
        #self.assertTrue(all(jnp.array(errors) < 10.0))

    #def test_ctd(self):
        #algo = CTD_Polynomial(rank=self.rank, key=self.key)
        #decoupling = algo.run(self.X, self.Y, self.J)
        #errors = function_error(target, decoupling, self.X, self.key)
        #self.assertTrue(all(jnp.array(errors) < 10.0))

    #def test_cmtf_bsd(self):
        #algo = CMTF_BSpline(rank=self.rank, key=self.key)
        #decoupling = algo.run(self.X, self.Y, self.J)
        #errors = function_error(target, decoupling, self.X, self.key)
        #self.assertTrue(all(jnp.array(errors) < 10.0))

    #def test_cmtf_psd(self):
        #algo = CMTF_PSpline(rank=self.rank, key=self.key)
        #decoupling = algo.run(self.X, self.Y, self.J)
        #errors = function_error(target, decoupling, self.X, self.key)
        #self.assertTrue(all(jnp.array(errors) < 10.0))

    def test_algorithm(self):
        algo = Algorithm(self.rank, 10, 5, self.key, 0.1, 5, 3)
        decoupling = algo.run(self.X, self.Y, self.J)
        errors = function_error(target, decoupling, self.X, self.key)
        print(errors)
        self.assertTrue(all(jnp.array(errors) < 10.0))
