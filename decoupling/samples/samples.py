from decoupling.types import *

from jaxtyping import jaxtyped
from beartype import beartype

class Samples:
    inputs: X_dtype
    outputs: Y_dtype
    jacobians: J_dtype

    @jaxtyped(typechecker=beartype)
    def __init__(self, inputs: X_dtype, outputs: Y_dtype, jacobians: J_dtype):
        self.inputs = inputs
        self.outputs = outputs
        self.jacobians = jacobians

    def unpack(self) -> Tuple[X_dtype, Y_dtype, J_dtype]:
        return (self.inputs, self.outputs, self.jacobians)

    def __str__(self) -> str:
        return f'Samples [X: {self.inputs.shape}, Y: {self.outputs.shape}, J: {self.jacobians.shape}]'
