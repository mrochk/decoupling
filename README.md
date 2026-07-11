![Logo](logo.png)

# Decoupling

Fast tensor decoupling in Jax.

This library is a collection of algorithms for decoupling multivariate functions using tensor decompositions.

```python
from decoupling.algorithm import BasicDecoupling
from decoupling.utils import collect_information, function_error

def target(x): # define a simple polynomial
    return jnp.array([x[0]**3 + x[1]**2 + x[0]*b, x[1]**3 + x[0]**2 + x[0]*b])

rank, N = 4, 30 # rank and number of samples
info = collect_information(target, N, key) # collect outputs and jacobians

decoupling = BasicDecoupling(rank, key=key).run(*info) # compute decoupling
errors = function_error(target, decoupling, info[0], key) # evaluate
```

This project was built using `uv` (https://docs.astral.sh/uv). 

### Installation

You can easily get `decoupling` from PyPI:
```bash
pip install decoupling
```
Otherwise, for a local installation:
```bash
git clone git@github.com:mrochk/decoupling.git
pip install decoupling
```

### Methodology

Tensor decoupling algorithms are used to find a decoupled representation of a target multivariate function. This is illustrated below.

<p align="center">
<img src="examples/images/decoupling.png" width=800>
</p>

In fact, this representation is a 2-layer MLP, meaning that tensor decoupling could be used to compress or build neural networks.

You can read about the basic methodology in this paper: https://arxiv.org/abs/1410.4060.

The goal of this library is to be the reference implementation of tensor decoupling algorithms. Our goal is to keep the source code as simple as possible, while being fast by leveraging Jax's JIT compiler, and, later, GPUs. 

The other important aspect is that it should be easy to design and add new algorithms, by leveraging already written and reusable code.

### Algorithms Implemented

- Polynomial Tensor Decoupling `decoupling/algorithm/basic` [Dreesen, Ishteva & Schoukens (2015)]
- Constrained Polynomial TD `decoupling/algorithm/ctd_polynomial` [Hollander, (2017)]
- CMTF B-Spline Decoupling `decoupling/algorithm/cmtf_bspline` [De Jonghe & Ishteva (2025)]
- CMTF P-Spline Decoupling `decoupling/algorithm/cmtf_pspline`

### Testing

```bash
uv run -m unittest discover testing -v # or ./test.sh
```
