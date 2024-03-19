import torch
from utils import find_onehot_dimension
from typing import Optional, Union, Callable


def per_label_dice(
    input: torch.Tensor,
    target: torch.Tensor,
    axes: Optional[Union[list, tuple]] = None,
    input_is_batch: bool = False,
    eps: float = 1e-8,
):

    assert input.shape == target.shape, "Input and target must be the same shape."
    assert find_onehot_dimension(input) is not None
    assert find_onehot_dimension(target) is not None

    if axes is None:
        if input_is_batch:
            spatial_dims = list(range(2, input.dim()))
        else:
            spatial_dims = list(range(1, input.dim()))

    intersection = torch.sum(input * target, dim=spatial_dims)  # intersection
    size_input = torch.sum(input, dim=spatial_dims)
    size_target = torch.sum(target, dim=spatial_dims)

    dice = 2 * intersection / (size_input + size_target + eps)
    dice[(intersection == 0) & ((size_input + size_target) == 0)] = 1

    if input_is_batch:
        return dice.mean(dim=0)
    return dice


def pairwise_jaccard_distance(
    A: torch.Tensor, B: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:

    assert find_onehot_dimension(A) is not None
    assert find_onehot_dimension(B) is not None

    B = B.transpose(1, 5)
    intersection = torch.sum(A & B, dim=(3, 4))  # B x N x C x M
    union = torch.sum(A | B, dim=(3, 4))
    pairwise_jaccard_distances = 1 - (intersection / (union + eps))

    # Union 0 means both A and B are 0, which actually means a perfect prediction
    pairwise_jaccard_distances[(union == 0) & (intersection == 0)] = 1

    # Move channels to the beginning
    pairwise_jaccard_distances = pairwise_jaccard_distances.transpose(1, 2)

    return pairwise_jaccard_distances  # B x C x N x M


def generalised_energy_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    metric: Callable = pairwise_jaccard_distance,
) -> torch.Tensor:
    """
    Calculate the (generalised) energy distance (https://en.wikipedia.org/wiki/Energy_distance)
    where x,y are torch.Tensors containing samples of the distributions to be
    compared for a given metric. The input can either be an array of samples (N x C x Sx x Sy)
    or a batch of arrays of samples (B x N x C x Sx x Sy). In the former case the function returns
    a GED value per class (C). In the later case it returns a matrix of GED values of shape B x C.
        Parameters:
            x (torch.Tensor): One set of N samples
            y (torch.Tensor): Another set of M samples
            metric (function): a function implementing the desired metric
        Returns:
            The generalised energy distance of the two samples (float)
    """

    assert x.dim() == y.dim()
    if x.dim() == 4:
        input_is_batch = False
    elif x.dim() == 5:
        input_is_batch = True
    else:
        raise ValueError(
            f"Unexpected dimension of input tensors: {x.dim()}. Expected 4 or 5."
        )

    assert find_onehot_dimension(x) is not None
    assert find_onehot_dimension(y) is not None

    if not input_is_batch:
        x, y = x.unsqueeze(0), y.unsqueeze(0)

    def expectation_of_difference(a, b):

        N, M = a.shape[1], b.shape[1]
        A = torch.tile(
            a[:, :, :, :, :, None], (1, 1, 1, 1, 1, M)
        )  # B x N x C x Sx x Sy x M
        B = torch.tile(
            b[:, :, :, :, :, None], (1, 1, 1, 1, 1, N)
        )  # B x M x C x Sx x Sy x N
        return metric(A, B).mean(dim=(2, 3))

    Exy = expectation_of_difference(x, y)
    Exx = expectation_of_difference(x, x)
    Eyy = expectation_of_difference(y, y)

    ed = torch.sqrt(2 * Exy - Exx - Eyy)

    if not input_is_batch:
        ed = ed[0]

    return ed
