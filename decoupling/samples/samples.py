from beartype import beartype
from jaxtyping import jaxtyped

from decoupling.types import (
    inputs_dtype,
    outputs_dtype,
    jacobians_dtype,
)

class Samples:

    @jaxtyped(typechecker=beartype)
    def __init__(
        self,
        inputs: inputs_dtype,
        outputs: outputs_dtype,
        jacobians: jacobians_dtype
    ):

        self.inputs = inputs
        self.outputs = outputs
        self.jacobians = jacobians

    def __iter__(self):
        yield self.inputs
        yield self.outputs
        yield self.jacobians
