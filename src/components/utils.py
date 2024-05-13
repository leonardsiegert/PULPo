import torch
from src.network_blocks import ResizeTransform


# resize a dict of dfs to the size of the first df or a target size
def resize_dfs(dfs: dict[int, torch.Tensor], target_size: list[int] = None) -> dict[int, torch.Tensor]:
    """ Resize a dict of deformation fields to the size of the first deformation field or to a target size."""
    scaled_dfs = {}
    for l in range(dfs.keys()):
        if target_size == None:
            resizer = ResizeTransform(vel_resize = 1 / (dfs[0].size()[0] / dfs[l].size()[0]), ndims = len(dfs[l].size()[2:]))
        else:
            resizer = ResizeTransform(vel_resize = 1 / (target_size[0] / dfs[l].size()[0]), ndims = len(target_size[2:]))
        scaled_dfs[l] = resizer(dfs[l])
    return scaled_dfs

def warp_landmarks(lm: torch.Tensor, df:torch.Tensor) -> torch.Tensor:
    """ Warp landmarks using a deformation field.
        Args:
            lm (torch.Tensor): Landmarks to be warped.
                        Shape: (1, num_landmarks, ndims)
            df (torch.Tensor): Deformation field.
                        Shape: (1, ndims, H, W, D)
    """
    lm = lm.long()
    new_lm = lm - df[:,:,lm[0,:,0],lm[0,:,1],lm[0,:,2]].transpose(-2,-1)
    return new_lm