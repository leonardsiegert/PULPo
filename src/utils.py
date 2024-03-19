from typing import Iterable, Tuple, Optional, Mapping
import torch.nn as nn


def key_to_string(func):
    def inner(cls, *args):
        return func(cls, str(args[0]), *args[1:])

    return inner


def iterator_key_to_int(func):
    def _key_to_int(iter):
        for el in iter:
            if len(el) == 1:
                out = int(el[0])
            else:
                out = (int(el[0]),) + el[1:]
            yield out

    def inner(cls):
        return _key_to_int(func(cls))

    return inner


class ModuleIntDict(nn.ModuleDict):
    def __init__(self, modules: Optional[Mapping[int, nn.Module]] = None) -> None:
        super().__init__(modules)  # type: ignore[arg-type]

    @key_to_string
    def __getitem__(self, key: int) -> nn.Module:
        return super().__getitem__(key)

    @key_to_string
    def __setitem__(self, key: int, module: nn.Module) -> None:
        return super().__setitem__(key, module)  # type: ignore[index]

    @key_to_string
    def __delitem__(self, key: int) -> None:
        return super().__delitem__(key)  # type: ignore[arg-type]

    @iterator_key_to_int
    def keys(self) -> Iterable[int]:
        return super().keys()

    @iterator_key_to_int
    def items(self) -> Iterable[Tuple[int, nn.Module]]:
        return super().items()
