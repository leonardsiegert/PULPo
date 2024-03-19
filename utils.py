import torch
import torch.nn.functional as F
from typing import Union


def convert_to_onehot(
    labels: torch.Tensor, num_classes: int, channel_dim: int = 1
) -> torch.Tensor:

    out = F.one_hot(labels.long(), num_classes)
    out = out.unsqueeze(channel_dim).transpose(channel_dim, out.dim())
    return out.squeeze(-1)


def find_onehot_dimension(T: torch.Tensor) -> Union[int, None]:

    if float(T.max()) <= 1 and float(T.min()) >= 0:
        for d in range(T.dim()):
            if torch.all((T.sum(dim=d) - 1) ** 2 <= 1e-5):
                return d

    # Otherwise most likely not a one-hot tensor
    return None


def harden_softmax_outputs(T: torch.Tensor, dim: int) -> torch.Tensor:

    num_classes = T.shape[dim]
    out = torch.argmax(T, dim=dim)
    return convert_to_onehot(out, num_classes=num_classes, channel_dim=dim)


if __name__ == "__main__":

    label_batch = torch.randint(0, 3, size=(32, 64, 64))
    label_onehot = convert_to_onehot(label_batch, num_classes=3)
    label_softmax = label_onehot.float() + 0.01 * torch.rand(label_onehot.shape)
    label_softmax = F.softmax(label_softmax)
    label_hard = harden_softmax_outputs(label_softmax, dim=1)

    assert torch.all(label_hard == label_onehot)
