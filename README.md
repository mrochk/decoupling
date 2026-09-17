![Logo](logo.png)

# `Decoupling`: Fast tensor decoupling in Jax.

Tensor decoupling is a methodology for decoupling multivariate functions using tensor decompositions.

This library aims to be the reference implementation of tensor decoupling algorithms.

### Usage

```python
import jax, jax.numpy as jnp
from decoupling import Algorithm
from decoupling.utils import (
    function_error_from_callable,
    collect_information_from_random,
)

def target(x): # define a simple polynomial
    return jnp.array([
        x[0]**3 + x[1]**2 + x[0]*x[1],
        x[1]**3 + x[0]**2 + x[0]*x[1],
    ])

key = jax.random.key(0)
rank = 4 

# collect information (inputs, outputs, jacobians)
info = collect_information_from_random(target, N=100, key=key, ninputs=2)

decoupling = Algorithm(rank, key).run(*info) # compute decoupling

e1, e2 = function_error_from_callable(target, decoupling, info[0])
print(f'Errors: e1={e1:.2f}, e2={e2:.2f}') # compare to target
```

### Installation

You can easily get `decoupling` from PyPI:
```bash
pip install decoupling
```
Otherwise:
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

You can read more about the basic methodology in this paper: https://arxiv.org/abs/1410.4060.

Our goal is to keep the source code as simple as possible, while being fast by leveraging Jax's JIT compiler. 

### Examples

To learn more about how this library works, feel free to take a look at some [examples](./examples/).

### Testing

```bash
uv run python -m pytest -s # or ./test.sh
```
