import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from src.custom_types import OutputDictType, ReconstructionLossType, KLDivergenceType


class KL_nondiagonal():
    # works with shape [batch_size,3,H,W,D] or [batch_size,2,H,W]
    def __init__(self,inshape, prior_lambda=20) -> None:
        self.prior_lambda = prior_lambda # 20 pls
        self.inshape = inshape
        self.ndims = len(inshape)
        self.prodsize = torch.prod(torch.tensor(inshape))
        print("prodsize: ", self.prodsize)
        Conv = getattr(F, 'conv%dd' % self.ndims)
        # create Degree matrix
        ones = torch.ones((1,1,*inshape)) # ones has shape [1,1,H,W,D]
        kernel = (3,3) if self.ndims == 2 else (3,3,3)
        sum_filt = torch.ones((1,1,*kernel), requires_grad=False)
        # D has shape [batch_size, 1, H, W, D]
        self.D = Conv(ones,sum_filt,bias=None,stride=1,padding=1) -1
        self.D = self.D.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("device in KL: ", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        print("D device: ", self.D.device)
    
    def precision_loss(self, flow_mean):
        sm = 0
        for i in range(self.ndims):
            d = i+2
            r = [0,1,*range(d,self.ndims+2), *range(d-i,d)] # all 3 permutations we need
            m = flow_mean.permute(r)
            # calculates gradient in each direction
            df = m[:,:,1:,...] - m[:,:,:-1,...] # the sliced dim will be H,W,D in the 3 loop iterations
            sm += torch.mean(df*df) # mean over all spatial and batch dims
        return 0.5 * sm / self.ndims


    def loss(self, prior_mean, prior_sigma, flow_mean, flow_sigma):
        """Returns KL[p0 || p1], where p1 is a naive prior N(0,1)"""
        # flow_logsigma entspricht dem log der Covarianzmatrix
        # unser sigma ist das kleine Sigma, also müssen wir log(sigma^2) nehmen, um flow_sigma zu machen
        # our sigma is sigma, so we need to take log(sigma^2) to make flow_sigma
        flow_sigma = flow_sigma**2
        # compute the trace of the covariance matrix
        sigma_term = self.prior_lambda * self.D * flow_sigma - torch.log(flow_sigma)
        # compute the precision/mu part
        precision_term = (self.prior_lambda/2) * self.precision_loss(flow_mean)
        # combine the two terms
        return (torch.mean(sigma_term) + precision_term) * self.ndims * 0.5 * self.prodsize


def KL_two_gauss_with_diag_cov(
    mu0: torch.Tensor,
    sigma0: torch.Tensor,
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:

    """Returns KL[p0 || p1]"""

    sigma0_fs = torch.square(torch.flatten(sigma0, start_dim=1))
    sigma1_fs = torch.square(torch.flatten(sigma1, start_dim=1))

    logsigma0_fs = torch.log(sigma0_fs + eps)
    logsigma1_fs = torch.log(sigma1_fs + eps)

    mu0_f = torch.flatten(mu0, start_dim=1)
    mu1_f = torch.flatten(mu1, start_dim=1)

    return torch.mean(
            0.5
            * torch.sum(
                torch.div(
                    sigma0_fs + torch.square(mu1_f - mu0_f),
                    sigma1_fs + eps,
                )
                + logsigma1_fs
                - logsigma0_fs
                - 1,
                dim=1,
            )
        )
    # with naive prior:
    # sigma0_fs = torch.square(torch.flatten(sigma0, start_dim=1))
    # logsigma0_fs = torch.log(sigma0_fs + eps)
    # mu0_f = torch.flatten(mu0, start_dim=1)
    # return torch.mean(0.5 * torch.sum(sigma0_fs + torch.square(mu0_f) - logsigma0_fs - 1, dim=1))


def L2_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    criterion = torch.nn.MSELoss(reduction="none")
    input_size = input.size()[2:]
    sumdims = list(range(2, len(input_size) + 2))
    return torch.mean(torch.sum(criterion(input=input, target=target),dim=sumdims))

def NCC_loss(y_pred, y_true, win_size=9, ncc_factor=100):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        # TODO: No, mine are shaped [batch_size, nb_feats, *vol_shape]
        ndims = len(Ii.size()[2:])
        assert ndims in [2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = [win_size] * ndims
        
        # compute filters
        sum_filt = torch.ones([1, 1, *win], device=Ii.device, requires_grad=False)

        pad_no = win[0] // 2

        if ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        Conv = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = Conv(Ii, sum_filt, stride=stride, padding=padding)
        J_sum = Conv(Ji, sum_filt, stride=stride, padding=padding)
        I2_sum = Conv(I2, sum_filt, stride=stride, padding=padding)
        J2_sum = Conv(J2, sum_filt, stride=stride, padding=padding)
        IJ_sum = Conv(IJ, sum_filt, stride=stride, padding=padding)

        win_size = np.prod(win)
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc = cross * cross / (I_var * J_var + 1e-8)
        # mean over the batch dimension, sum over spatial dimensions
        cc = torch.mean(cc, dim=0)
        return -torch.sum(cc) / ncc_factor

def Soft_dice_loss(input: torch.Tensor, target: torch.Tensor, dice_factor=50) -> torch.Tensor:
    input_size = input.size()[2:]
    # the dimensions over which to sum
    sumdims = list(range(2, len(input_size) + 2))
    # the dimensions multiplied together
    prod_size = torch.prod(torch.tensor(input_size))
    epsilon = 1e-6
    dice = ((2. * target*input).sum(dim=sumdims) + epsilon) / ((target**2).sum(dim=sumdims) + (input**2).sum(dim=sumdims) + epsilon)
    return torch.mean(1 - dice) * prod_size / dice_factor

def jacobian_det(deformation_field: torch.Tensor, lamb=None, normalize=True) -> torch.Tensor:
    # expects shape B,2,(D),H,W
    # returns shape B,1,(D),H,W")
    ndims = len(deformation_field.shape[2:])
    device = deformation_field.device

    if ndims == 2:
        if normalize:
            deformation_field = torch.stack((deformation_field[:,0] *2/deformation_field.shape[-2],
                                            deformation_field[:,1] *2/deformation_field.shape[-1]),1)
        B,_,H,W = deformation_field.size()
        rep_x = nn.ReplicationPad2d((1,1,0,0)).to(device)
        rep_y = nn.ReplicationPad2d((0,0,1,1)).to(device)

        kernel_y = nn.Conv2d(2,2,(3,1),bias=False,groups=2).to(device)
        kernel_y.weight.data[:,0,:,0] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(2,1).to(device)
        kernel_x = nn.Conv2d(2,2,(1,3),bias=False,groups=2).to(device)
        kernel_x.weight.data[:,0,0,:] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(2,1).to(device)

        disp_field_vox = deformation_field.flip(1)*(torch.Tensor([H-1,W-1]).to(device).view(1,2,1,1)-1)/2
        grad_y = kernel_y(rep_y(disp_field_vox))
        grad_x = kernel_x(rep_x(disp_field_vox))

        jacobian = torch.stack((grad_y,grad_x),1)+torch.eye(2,2).to(device).view(1,2,2,1,1)
        jac_det = jacobian[:,0,0,:,:]*jacobian[:,1,1,:,:] - jacobian[:,1,0,:,:]*jacobian[:,0,1,:,:]
    elif ndims == 3:
        if normalize:
            deformation_field = torch.stack((deformation_field[:,0] *2/deformation_field.shape[-3],
                                            deformation_field[:,1] *2/deformation_field.shape[-2],
                                            deformation_field[:,2] *2/deformation_field.shape[-1]),1)
        
        B,_,D,H,W = deformation_field.size()
        rep_x = nn.ReplicationPad3d((1,1,0,0,0,0)).to(device)
        rep_y = nn.ReplicationPad3d((0,0,1,1,0,0)).to(device)
        rep_z = nn.ReplicationPad3d((0,0,0,0,1,1)).to(device)

        kernel_z = nn.Conv3d(3,3,(3,1,1),bias=False,groups=3).to(device)
        kernel_z.weight.data[:,0,:,0,0] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(3,1).to(device)
        kernel_y = nn.Conv3d(3,3,(1,3,1),bias=False,groups=3).to(device)
        kernel_y.weight.data[:,0,0,:,0] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(3,1).to(device)
        kernel_x = nn.Conv3d(3,3,(1,1,3),bias=False,groups=3).to(device)
        kernel_x.weight.data[:,0,0,0,:] = torch.tensor([-0.5,0,0.5]).view(1,3).repeat(3,1).to(device)

        disp_field_vox = deformation_field.flip(1)*(torch.Tensor([D-1,H-1,W-1]).to(device).view(1,3,1,1,1)-1)/2
        grad_z = kernel_z(rep_z(disp_field_vox))
        grad_y = kernel_y(rep_y(disp_field_vox))
        grad_x = kernel_x(rep_x(disp_field_vox))

        jacobian = torch.stack((grad_z,grad_y,grad_x),1)
        eye = torch.eye(3,3).to(device).view(1,3,3,1,1,1)
        jacobian = jacobian+eye
        jac_det = jacobian[:,0,0,:,:,:]*(jacobian[:,1,1,:,:,:]*jacobian[:,2,2,:,:,:]-jacobian[:,2,1,:,:,:]*jacobian[:,1,2,:,:,:]) - jacobian[:,0,1,:,:,:]*(jacobian[:,1,0,:,:,:]*jacobian[:,2,2,:,:,:]-jacobian[:,2,0,:,:,:]*jacobian[:,1,2,:,:,:]) + jacobian[:,0,2,:,:,:]*(jacobian[:,1,0,:,:,:]*jacobian[:,2,1,:,:,:]-jacobian[:,2,0,:,:,:]*jacobian[:,1,1,:,:,:])
    return jac_det


def JDetStd(deformation_field: torch.Tensor, lamb=0, normalize=True) -> torch.Tensor:
    return lamb * jacobian_det(deformation_field, normalize=normalize).std()


# Regularization loss for the deformation field based on the L2 norm of the gradient
def L2_reg(deformation_field: torch.Tensor, lamb=0) -> torch.Tensor:
    num_dims = len(deformation_field.size()[2:])
    if num_dims == 2:
        H,W = deformation_field.size()[-2:]
        distH = (deformation_field[:,:,1:,1:] - deformation_field[:,:,:-1,1:])**2
        distW = (deformation_field[:,:,1:,1:] - deformation_field[:,:,1:,:-1])**2
        # to get a picture, one could use the following:
        # torch.nn.functional.pad(distH + distW, (0, 1, 0, 1))
        return (distH+distW).mean() * lamb * H * W
    elif num_dims == 3:
        H,W,D = deformation_field.size()[-3:]
        distH = (deformation_field[:,:,1:,1:,1:] - deformation_field[:,:,:-1,1:,1:])**2
        distW = (deformation_field[:,:,1:,1:,1:] - deformation_field[:,:,1:,:-1,1:])**2
        distD = (deformation_field[:,:,1:,1:,1:] - deformation_field[:,:,1:,1:,:-1])**2
        return (distH+distW+distD).mean() * lamb * H * W * D


class HierarchicalKLLoss(nn.Module):
    def __init__(
        self,
        KL_divergence: KLDivergenceType,
        weight_dict: dict[int, float],
        similarity_pyramid: bool,
        level_sizes: dict[int,torch.Tensor] = None
    ) -> None:
        super().__init__()

        self.weight_dict = weight_dict
        if similarity_pyramid:
            for l in self.weight_dict.keys():
                self.weight_dict[l] = self.weight_dict[l] / 2**l
        self.KL_divergence = KL_divergence
        if KL_divergence == KL_nondiagonal:
            self.KL_divergence = {key:KL_nondiagonal(inshape=level_sizes[key]).loss for key in level_sizes.keys()}


    def forward(
        self,
        prior_mus: OutputDictType,
        prior_sigmas: OutputDictType,
        posterior_mus: OutputDictType,
        posterior_sigmas: OutputDictType,
    ):

        assert self.weight_dict.keys() == prior_mus.keys()
        assert (
            prior_mus.keys()
            == prior_sigmas.keys()
            == posterior_mus.keys()
            == posterior_sigmas.keys()
        )

        kl_loss = 0.0
        all_levels = {}
        for l, w in self.weight_dict.items():
            #check if the KL_divergence is a dict or a function
            if isinstance(self.KL_divergence, dict):
                all_levels[l] = w * self.KL_divergence[l](
                    prior_mus[l], prior_sigmas[l], posterior_mus[l], posterior_sigmas[l]
                )
            else:
                all_levels[l] = w * self.KL_divergence(
                    posterior_mus[l], posterior_sigmas[l], prior_mus[l], prior_sigmas[l]
                )
            kl_loss += all_levels[l]  # type: ignore[assignment]

        return kl_loss, all_levels


class HierarchicalReconstructionLoss(nn.Module):
    def __init__(
        self, recon_loss: list[str], weight_dict: dict[int, float], similarity_pyramid: bool,ndims: int, window_size: dict[int, float]
    ) -> None:
        super().__init__()

        self.recon_loss = recon_loss
        self.weight_dict = weight_dict
        # self.weight_dict_ncc = weight_dict.copy()
        latent_levels = len(weight_dict.keys())
        if similarity_pyramid:
            for l in self.weight_dict.keys():
                self.weight_dict[l] = self.weight_dict[l] / 2**l
                # self.weight_dict_ncc[l] = self.weight_dict_ncc[l] / 2**(latent_levels-1-l)
        self.window_size = window_size
        self.ndims = ndims
        if self.ndims == 3:
            self.mode = 'trilinear'
        else:
            self.mode = 'bilinear'

    def forward(
        self,
        y_hat: OutputDictType,
        y: torch.Tensor,
        y_hat_seg: OutputDictType = None,
        seg_y: torch.Tensor = None,
        ncc_factor: int = 100,
        dice_factor: int = 50,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[int, torch.Tensor]]]:
        loss: torch.Tensor = 0.0  # type: ignore[assignment]
        all_levels = {}
        for l, w in self.weight_dict.items():
            y_target = F.interpolate(y, size=y_hat[l].size()[2:], mode=self.mode, align_corners=False)
            # if y_hat_seg is not None and seg_y is not None:
            #     seg_target=F.interpolate(seg_y, size=y_hat_seg[l].size()[2:], mode=self.mode, align_corners=False)
            # else:
            #     seg_target=None
            all_levels[l] = 0.0
            if "mse" in self.recon_loss:
                all_levels[l] += w * L2_loss(y_hat[l], y_target)
            if "ncc" in self.recon_loss:
                all_levels[l] += w * NCC_loss(y_hat[l], y_target, ncc_factor=ncc_factor, win_size=self.window_size[l])
                # all_levels[l] += self.weight_dict_ncc[l] * NCC_loss(y_hat[l], y_target, ncc_factor=ncc_factor, win_size=self.window_size[l])
            if "dice" in self.recon_loss:
                seg_target=F.interpolate(seg_y, size=y_hat_seg[l].size()[2:], mode=self.mode, align_corners=False)
                all_levels[l] += w * Soft_dice_loss(y_hat_seg[l], seg_target, dice_factor=dice_factor)

            all_levels[l] = all_levels[l] / len(self.recon_loss)
            # all_levels[l] = w * combined_loss(recon_loss=self.recon_loss, seg_input=y_hat_seg[l],seg_target=seg_target,
            #                                   y_pred=y_hat[l], y_target=F.interpolate(y, size=y_hat[l].size()[2:], mode=self.mode, align_corners=False),
            #                                   ncc_factor=ncc_factor, win_size=self.window_size[l], dice_factor=dice_factor)
            loss += all_levels[l]
        return loss, all_levels
        
class HierarchicalRegularization(nn.Module):

    def __init__(
        self, regularizer, weight_dict: dict[int, float], similarity_pyramid: bool
    ) -> None:
        super().__init__()
        self.regularizer = regularizer
        self.weight_dict = weight_dict
        if similarity_pyramid:
            for l in self.weight_dict.keys():
                self.weight_dict[l] = self.weight_dict[l] / 2**l

    def forward(
        self,
        dfs: OutputDictType,
        lamb: float = 0,
    ) -> Union[torch.Tensor, tuple[torch.Tensor, dict[int, torch.Tensor]]]:
        
        assert self.weight_dict.keys() == dfs.keys()

        total_loss = 0.0
        all_levels = {}
        for l, w in self.weight_dict.items():
            all_levels[l] = w * self.regularizer(dfs[l], lamb)
            total_loss += all_levels[l]

        return total_loss, all_levels
    