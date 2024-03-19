import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


def gauss_sampler(mu: torch.Tensor, sigma: torch.Tensor, var: Optional[int]=1) -> torch.Tensor:
    return mu + sigma * (var * torch.randn_like(sigma, dtype=torch.float32))


class ConvUnit(nn.Module):
    def __init__(self, input_size: list[int], in_channels: int, out_channels: int = None) -> None:
        super().__init__()

        if not out_channels:
            out_channels = in_channels

        ndims = len(input_size)
        Conv = getattr(nn, 'Conv%dd' % ndims)
        BatchNorm = getattr(nn, 'BatchNorm%dd' % ndims)
        
        self._op = nn.Sequential(
            Conv(in_channels, out_channels, kernel_size=3, padding=1),
            BatchNorm(out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor):
        return self._op(x)


class ConvSequence(nn.Module):
    def __init__(
        self,
        input_size: list[int],
        in_channels: int,
        out_channels: int,
        depth: int,
    ) -> None:
        super().__init__()
        first_conv = ConvUnit(input_size, in_channels, out_channels)
        convs = [first_conv] + [ConvUnit(input_size, out_channels) for _ in range(depth - 1)]
        self._op = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor):
        return self._op(x)


class MuSigmaBlock(nn.Module):
    def __init__(self, input_size: list[int], in_channels: int, zdim: int) -> None:
        super().__init__()
        ndims = len(input_size)
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self._conv_mu = Conv(in_channels, zdim, kernel_size=1)
        # self._conv_mu.weight = nn.Parameter(torch.normal(mean=0.0,std=1e-5,size=self._conv_mu.weight.shape))
        # self._conv_mu.bias = nn.Parameter(torch.zeros(self._conv_mu.bias.shape))
        self._conv_sigma = nn.Sequential(
            Conv(in_channels, zdim, kernel_size=1),
            nn.Softplus(),
        )
        # self._conv_sigma[0].weight = nn.Parameter(torch.normal(mean=0.0,std=1e-10,size=self._conv_sigma[0].weight.shape))
        # # HACK: I guess this is roughly the idea of the VXM initialization
        # self._conv_sigma[0].bias = nn.Parameter(torch.zeros(self._conv_sigma[0].bias.shape)-2.0)

    def forward(self, x: torch.Tensor):
        return [self._conv_mu(x), self._conv_sigma(x)]
    

class ControlPoints(nn.Module):
    def __init__(
        self,
        input_size: list[int],
        zdim: int,
        max_channels: int,
        depth: int,
    ) -> None:
        super().__init__()
        ndims = len(input_size)
        Conv = getattr(nn, 'Conv%dd' % ndims)
        if depth == 1:
            convs = [Conv(zdim, ndims, kernel_size=3)]
        elif depth == 0:
            convs = [nn.Identity()]
        else:
            first_conv = ConvUnit(input_size, zdim, max_channels)
            convs = [first_conv] + [ConvUnit(input_size, max_channels, max_channels) for _ in range(depth - 2)]
            convs = convs + [Conv(max_channels, ndims, kernel_size=1)]
        self._op = nn.Sequential(*convs)

    def forward(self, x: torch.Tensor):
        return self._op(x)
    

class SpatialTransformer(nn.Module):
    def __init__(self, size, mode='bilinear'):
        super().__init__()
        self.size = size
        self.mode = mode
        # create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors, indexing='ij')
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.float32)
        self.register_buffer('grid', grid, persistent=True)

    def forward(self, df, moving_image):
        # new locations
        new_locs = self.grid + df

        # need to normalize grid values to [-1, 1] for resampler
        for i in range(len(self.size)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (self.size[i] - 1) - 0.5)

        # move channels dim to last position
        # due to new 'ij'/'xy' conventionality, channels need to be reversed
        # this is the same as applying .flip(-1)
        if len(self.size) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(self.size) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        # resample image
        resulting_image = nn.functional.grid_sample(moving_image, new_locs, mode="bilinear", padding_mode="border", align_corners=False)
        return resulting_image
    

class ResizeTransform(nn.Module):
    """
    Resize a transform, which involves resizing the vector field *and* rescaling it.
    """

    def __init__(self, vel_resize, ndims):
        super().__init__()
        self.factor = 1.0 / vel_resize
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode

    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = F.interpolate(x, align_corners=False, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x

        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = F.interpolate(x, align_corners=False, scale_factor=self.factor, mode=self.mode)

        # don't do anything if resize is 1
        return x

class DFAdder(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, df1, df2):
        df = df1 + df2
        return df
    
class VecInt(nn.Module):
    """
    Integrates a vector field via scaling and squaring.
    """

    def __init__(self, inshape, nsteps):
        super().__init__()

        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer(inshape)

    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec
    
class BSplineInterpolate(nn.Module):
    def __init__(self, size, order:Optional[int]=3, kernel_size:Optional[int]=7,stride:Optional[int]=1,padding:Optional[int]=3):
        super().__init__()
        # spatial dimensions
        self.size = size
        # how many times avg_pool is applied - does this make sense?
        self.order = order
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if len(self.size) == 2:
            self.avg_pool = nn.AvgPool2d(kernel_size, stride=stride, padding=padding)
            self.mode = 'bilinear'
        elif len(self.size) == 3:
            self.avg_pool = nn.AvgPool3d(kernel_size, stride=stride, padding=padding)
            self.mode = 'trilinear'

    def forward(self, control_points):
        x = F.interpolate(control_points, size=self.size,mode=self.mode,align_corners=False)
        ic(x.shape)
        for o in range(self.order):
            x = self.avg_pool(x)
            ic(x.shape)

        return x