from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import overload, Optional

from src.custom_types import SamplerType, OutputDictType

# The following contains some prototypes for priors, posteriors, and decoders
# The idea is to capture the general expected form of those components
# somewhere.


class AbstractPrior(ABC, nn.Module):

    sampler: SamplerType

    @abstractmethod
    def forward(
        self, x: torch.Tensor, y:Optional[torch.Tensor] = None,
    ) -> tuple[OutputDictType, OutputDictType, OutputDictType]:
        ...


class AbstractPosterior(ABC, nn.Module):

    sampler: SamplerType

    @abstractmethod
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[OutputDictType, OutputDictType, OutputDictType]:
        ...


class AbstractDecoder(ABC, nn.Module):
    @overload
    @abstractmethod
    def forward(self, z: OutputDictType) -> OutputDictType:
        ...

    # Methods like the probabilistic U-Net have a "skip connection", i.e.
    # the x gets forwarded directly to the decoder
    @abstractmethod
    def forward(self, z: OutputDictType, x: torch.Tensor = None) -> OutputDictType:
        ...
