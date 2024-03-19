import torch
from typing import Callable, Protocol

# Define Type for random sampling function. Implicit assumption:
# Samplers always are of type "mu, sigma \mapsto sample". May have
# to generalise in the future.
SamplerType = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

OutputDictType = dict[int, torch.Tensor]
KLDivergenceType = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
]


class ReconstructionLossType(Protocol):
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ...
