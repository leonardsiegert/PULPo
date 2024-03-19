import torch
from torchvision.utils import make_grid, draw_segmentation_masks, flow_to_image  # type: ignore[import]
import torch.nn.functional as F
torch.set_float32_matmul_precision("medium") # this possibly speeds up computations without losing much precision 
import math
from typing import Optional
from abc import ABC, abstractmethod
import pytorch_lightning as pl
import numpy as np
import time

from src.losses import (
    HierarchicalKLLoss,
    HierarchicalReconstructionLoss,
    HierarchicalRegularization,
    Soft_dice_loss,
    L2_loss,
    NCC_loss,
    jacobian_det,
    JDetStd,
    L2_reg,
    KL_two_gauss_with_diag_cov,
    KL_nondiagonal,
)
from src.custom_types import SamplerType, OutputDictType
from src.network_blocks import gauss_sampler, SpatialTransformer, ResizeTransform, DFAdder, VecInt
from src.base import AbstractPrior, AbstractPosterior, AbstractDecoder
from src.metrics import per_label_dice, generalised_energy_distance
from utils import convert_to_onehot, find_onehot_dimension, harden_softmax_outputs


class AbstractHierarchicalProbabilisticModel(ABC, pl.LightningModule):
    """
    This is the base class for all conditional hierarchical models. This
    can be extended also for other use cases then segmentation.
    """

    prior: AbstractPrior
    posterior: AbstractPosterior
    decoder: AbstractDecoder

    hierarchical_kl_loss: HierarchicalKLLoss
    hierarchical_recon_loss: HierarchicalReconstructionLoss
    hierarchical_regularization: HierarchicalRegularization

    total_levels: int
    latent_levels: int

    validation_counter = 0

    def predict_output_samples(self, x: torch.Tensor, y: torch.Tensor, N: int = 1) -> tuple[torch.Tensor,torch.Tensor]:
        bs = x.shape[0]
        xb = torch.vstack([x for _ in range(N)])
        yb = torch.vstack([y for _ in range(N)])
        down_activations = self.downpath(xb, yb)
        posterior_mus, posterior_sigmas, z, control_points, individual_dfs, combined_dfs, final_dfs, outputs = self.autoencoder(xb, down_activations)
        # for key in posterior_sigmas:
        #     print(f"posterior_sigmas {key} min: ", posterior_sigmas[key].min())
        #     print(f"posterior_sigmas {key} max: ", posterior_sigmas[key].max())
        #     print(f"posterior_sigmas {key} avg: ", posterior_sigmas[key].mean())
        #     print(f"posterior_mus {key} min: ", posterior_mus[key].min())
        #     print(f"posterior_mus {key} max: ", posterior_mus[key].max())
        #     print(posterior_mus[key].unique())
        # reshape them from BxN,C,H,W(,D) to B,N,C,H,W(,D)
        outputs_reshaped = {key:outputs[key].view([N, bs]+[*outputs[key].shape][1:]).transpose(0,1) for key in outputs}
        individual_dfs_reshaped = {key:individual_dfs[key].view([N, bs]+[*individual_dfs[key].shape][1:]).transpose(0,1) for key in individual_dfs}
        return outputs_reshaped, individual_dfs_reshaped

class AbstractHierarchicalProbabilisticRegistrationModel(
    AbstractHierarchicalProbabilisticModel
):
    """
    This is the base class for all hierarchical probabilistic registration models such as PHIReg
    """

    # This should average over the dfs from N samples
    # and then apply the average df to the moving image
    # should return outputs, dfs
    
    # resize a dict of dfs to the size of the first df or a target size
    def resize_dfs(self, dfs: OutputDictType, target_size: list[int] = None) -> OutputDictType:
        scaled_dfs = {}
        for l in range(dfs.keys()):
            if target_size == None:
                resizer = ResizeTransform(vel_resize = 1 / (dfs[0].size()[0] / dfs[l].size()[0]), ndims = len(dfs[l].size()[2:]))
            else:
                resizer = ResizeTransform(vel_resize = 1 / (target_size[0] / dfs[l].size()[0]), ndims = len(target_size[2:]))
            scaled_dfs[l] = resizer(dfs[l])
        return scaled_dfs

    # combine individual level dfs into the combined dfs
    def combine_dfs(self, individual_dfs: OutputDictType) -> OutputDictType:
            combined_dfs, final_dfs = {}, {}
            for l in reversed(range(self.latent_levels)):
                if l+1 in combined_dfs:
                    # resize the df to the size of the next level
                    resizer = ResizeTransform(vel_resize = 1 / (individual_dfs[l].size()[2] / individual_dfs[l+1].size()[2]), ndims = len(self.input_size))
                    combined_dfs[l] = self.df_combiner(individual_dfs[l], resizer(combined_dfs[l+1]))
                else:
                    combined_dfs[l] = individual_dfs[l]
            if self.hparams.decoder == "BSpline":
                final_dfs = combined_dfs
            elif self.hparams.decoder == "SVF":
                for l in reversed(range(self.latent_levels)):
                    # integrate
                    integrate = VecInt(combined_dfs[l].size()[2:],nsteps=7)
                    integrate = integrate.to(combined_dfs[l].device)
                    final_dfs[l] = integrate(combined_dfs[l])
                    # resize
                    # HACK: 
                    target_size = self.input_size if (l == 0 or self.hparams.df_resolution == "full_res") else combined_dfs[l].size()[2:]
                    # target_size = self.input_size if (l == 0 or self.hparams.df_resolution == "full_res") else combined_dfs[l-1].size()[2:]
                    resizer = ResizeTransform(vel_resize = 1 / (target_size[0] / final_dfs[l].size()[2]), ndims = len(self.input_size))
                    final_dfs[l] = resizer(final_dfs[l])
            else:
                raise ValueError(f"Hyperparameter decoder is {self.hparams.decoder}. Not a known option.")
        
            return combined_dfs, final_dfs


    def predict(self, x: torch.Tensor, y: torch.Tensor, N: int = 1) -> torch.Tensor:
        _, individual_dfs = self.predict_output_samples(x, y, N)
        # average the dfs over N
        avg_dfs = {key:individual_dfs[key].mean(dim=1) for key in individual_dfs}
        # combine the average dfs
        avg_combined_dfs, avg_final_dfs = self.combine_dfs(avg_dfs)
        # apply the combined df to the moving image
        avg_outputs = {key: self.autoencoder.decoders[key].spatial_transform(avg_final_dfs[key], x) for key in avg_final_dfs}
        return avg_outputs, avg_dfs
    
    def predict_deterministic(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        down_activations = self.downpath(x, y)
        mu, _, z, control_points, individual_dfs, combined_dfs, final_dfs, outputs = self.autoencoder(x, down_activations, deterministic=True)
        return outputs, individual_dfs


    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        down_activations = self.downpath(x, y)
        _, _, _, _, _, _, _, outputs = self.autoencoder(x, down_activations)
        return outputs[0]

    def transform_segmentation(self, dfs: OutputDictType, seg: torch.Tensor) -> OutputDictType:
        # dictionary for the segmentations on all levels but the original
        # dictionary for x on the scale of the latent levels. highest level having original size
        ndims = len(seg.shape[2:])
        avgPool = getattr(F, 'avg_pool%dd' % ndims)
        
        if self.df_resolution == "full_res":
            level_seg = {l: seg for l in range(self.latent_levels)}
        else:
            level_seg = {0: seg}
            # bring it to the size of the first latent level
            # HACK: hardcoding the resolution thing
            for k in range(self.lk_offset):
            # for k in range(self.lk_offset-1):
                level_seg[0] = avgPool(level_seg[0], kernel_size=2, stride=2, padding=0, ceil_mode=True)
            # bring it to the size of all latent levels
            for l in range(1, self.latent_levels):
                level_seg[l] = avgPool(level_seg[l-1], kernel_size=2, stride=2, padding=0, ceil_mode=True)
            # set the original image on the lowest level
            level_seg[0] = seg
        transformed_segs = {key: self.autoencoder.decoders[key].spatial_transform(dfs[key], level_seg[key]) for key in dfs}
        return transformed_segs

    def warp_landmarks(self, lm: torch.Tensor, df:torch.Tensor) -> torch.Tensor:
        # look up the displacement field at the landmark coordinates
        # expects lm to be of shape (1, num_landmarks, ndims)
        # expects df to be of shape (n_samples, ndims, H, W, D)
        lm = lm.long()
        # TODO: Why do I substract the df? Shouldn't I add it?
        # new_lm = lm + df[:,:,lm[0,:,0],lm[0,:,1],lm[0,:,2]].transpose(-2,-1)
        new_lm = lm - df[:,:,lm[0,:,0],lm[0,:,1],lm[0,:,2]].transpose(-2,-1)
        return new_lm

    def training_step(self, batch, batch_idx):
        x,y,seg_x,seg_y,lm1,lm2,mask1,mask2 = batch

        down_activations = self.downpath(x,y)
        posterior_mus, posterior_sigmas, posterior_samples, control_points, individual_dfs, combined_dfs, final_dfs, y_hat = self.autoencoder(x, down_activations)
        prior_mus, prior_sigmas = self.prior(posterior_mus, posterior_sigmas)
        
        # if we are using segmentations, transform them
        if "dice" in self.hparams.recon_loss:
            y_hat_seg = self.transform_segmentation(final_dfs, seg_x)
        else:
            y_hat_seg = {key: None for key in final_dfs}

        # st = time.time()
        kl_loss, kl_loss_levels = self.hierarchical_kl_loss(
            prior_mus,
            prior_sigmas,
            posterior_mus,
            posterior_sigmas,
        )
        # print("kl loss time: ", time.time()-st)
        kl_loss = kl_loss * self.beta
        kl_loss_levels.update({n: self.beta * kl_loss_levels[n] for n in kl_loss_levels.keys()})

        # st = time.time()
        reconstruction_loss, reconstruction_loss_levels = self.hierarchical_recon_loss(
            y_hat, y, y_hat_seg, seg_y, ncc_factor=self.hparams.ncc_factor, dice_factor=self.hparams.dice_factor)
        # print("recon loss time: ", time.time()-st)
        # st = time.time()
        regularization_loss, regularization_loss_levels = self.hierarchical_regularization(
            final_dfs,lamb=self.hparams.lamb)
        # print("regularization loss time: ", time.time()-st)
        total_loss = kl_loss + reconstruction_loss + regularization_loss
        
        ic(reconstruction_loss, reconstruction_loss_levels)
        ic(kl_loss,kl_loss_levels)
        ic(regularization_loss, regularization_loss_levels)
        ic(total_loss)

        self.log("train/kl_loss", kl_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/reconstruction_loss", reconstruction_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/regularization_loss", regularization_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/total_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True)

        for level in kl_loss_levels.keys():
            self.log(
                f"train_levels/kl loss level {level}",
                kl_loss_levels[level],
                on_step=False,
                on_epoch=True
            )
            self.log_dict({f"train_distribution_levels/mean_prior_mu_{level}":torch.mean(prior_mus[level]),
                f"train_distribution_levels/mean_prior_sigma_{level}":torch.mean(prior_sigmas[level]),
                f"train_distribution_levels/mean_posterior_mu_{level}":torch.mean(posterior_mus[level]),
                f"train_distribution_levels/mean_posterior_sigma_{level}":torch.mean(posterior_sigmas[level])
                }, on_step=False, on_epoch=True)

        for level in reconstruction_loss_levels.keys():
            self.log(
                f"train_levels/recon loss level {level}",
                reconstruction_loss_levels[level],
                on_step=False,
                on_epoch=True
            )
        
        for level in regularization_loss_levels.keys():
            self.log(
                f"train_levels/regularization loss level {level}",
                regularization_loss_levels[level],
                on_step=False,
                on_epoch=True
            )
            if torch.sum(torch.isnan(regularization_loss_levels[level])) > 0:
                print("NAN IN REGULARIZATION LOSS")
                # save the state dict
                torch.save(self.state_dict(), "nan_state_dict.pt")
                # break the training loop
                self.trainer.should_stop = True


        return total_loss

    def validation_step(self, batch, batch_idx):
        # Increase the counter at the end of each validation epoch
        if batch_idx == self.trainer.num_val_batches[0]-1:
            self.validation_counter +=1

        x,y,seg_x,seg_y,lm1,lm2,mask1,mask2 = batch

        down_activations = self.downpath(x,y)
        posterior_mus, posterior_sigmas, posterior_samples, control_points, individual_dfs, combined_dfs, final_dfs, y_hat = self.autoencoder(x, down_activations)
        prior_mus, prior_sigmas = self.prior(posterior_mus, posterior_sigmas)

        # if we are using segmentations, transform them
        if "dice" in self.recon_loss:
            y_hat_seg = self.transform_segmentation(final_dfs, seg_x)
        else:
            y_hat_seg = {key: None for key in final_dfs}
            

        kl_loss, kl_loss_levels = self.hierarchical_kl_loss(
            prior_mus,
            prior_sigmas,
            posterior_mus,
            posterior_sigmas,
        )

        kl_loss = kl_loss * self.beta
        kl_loss_levels.update({n: self.beta * kl_loss_levels[n] for n in kl_loss_levels.keys()})
                
        reconstruction_loss, reconstruction_loss_levels = self.hierarchical_recon_loss(
            y_hat, y, y_hat_seg, seg_y, ncc_factor=self.hparams.ncc_factor, dice_factor=self.hparams.dice_factor)
        
        regularization_loss, regularization_loss_levels = self.hierarchical_regularization(
            final_dfs,lamb=self.hparams.lamb)
        ic(regularization_loss, regularization_loss_levels)

        ic(kl_loss, reconstruction_loss)
        total_loss = kl_loss + reconstruction_loss + regularization_loss

        self.log("val/kl_loss", kl_loss, on_epoch=True)
        self.log("val/reconstruction_loss", reconstruction_loss, on_epoch=True)
        self.log("val/regularization_loss", regularization_loss, on_epoch=True)
        self.log("val/total_loss", total_loss, on_epoch=True)

        for level, level_loss in reconstruction_loss_levels.items():
            self.log(f"val_levels/recon loss level {level}", level_loss, on_epoch=True)
        for level, level_loss in kl_loss_levels.items():
            self.log(f"val_levels/kl loss level {level}", level_loss, on_epoch=True)

        for level, level_loss in regularization_loss_levels.items():
            self.log(f"val_levels/kl loss level {level}", level_loss, on_epoch=True)

        for level in kl_loss_levels.keys():
            self.log(
                f"val_distribution_levels/mean_prior_mu_{level}",
                torch.mean(prior_mus[level]),
            )
            self.log(
                f"val_distribution_levels/mean_prior_sigma_{level}",
                torch.mean(prior_sigmas[level]),
            )
            self.log(
                f"val_distribution_levels/mean_posterior_mu_{level}",
                torch.mean(posterior_mus[level]),
            )
            self.log(
                f"val_distribution_levels/mean_posterior_sigma_{level}",
                torch.mean(posterior_sigmas[level]),
            )

        # For the costly image logging, we don't want to log every epoch for small validation sets.
        # If the amount of validation batches is low, log images every 'image_logging_frequency' validation steps instead 
        if self.trainer.num_val_batches[0] < self.hparams.image_logging_frequency:
            condition = self.validation_counter % self.hparams.image_logging_frequency == 0
        else:
            # The end of each validation epoch
            condition = batch_idx == self.trainer.num_val_batches[0]-1

        if condition:
            y_pred = y_hat[0]
            df = final_dfs[0]
            # absolute distance calculation that works with min-max normalization
            distance = (y_pred-y+1)/2
            distance = torch.where(distance>1, 1., distance)

            # if batch size >9, the image grids will look bad, in that case only consider first 9 images
            if x.size()[0] > 9:
                x = x[:9]
                y = y[:9]
                y_pred = y_pred[:9]
                distance = distance[:9]
                df = df[:9]
                for l in range(self.latent_levels):
                    individual_dfs[l] = individual_dfs[l][:9]
                    final_dfs[l] = final_dfs[l][:9]
                    y_hat[l] = y_hat[l][:9]

            # Log all images
            for name, img in zip(
                ["val/x", "val/y", "val/y_pred", "val/distance", "val/DF"],
                [x, y, y_pred, distance, df],
            ):
                if self.ndims == 3:
                    img = img[:,:,:,img.shape[-2]//2,:]
                if name == "val/DF":
                    # cut out the dimension that we are slicing
                    img = torch.stack([img[:,0,:,:], img[:,self.ndims-1,:,:]], dim=1)
                    img = flow_to_image(img)
                self._log_images_in_grid(img, name)

            for l in range(self.latent_levels):
                if self.ndims == 3:
                    y_hat[l] = y_hat[l][:,:,:,y_hat[l].shape[-2]//2,:]
                    # cut out the dimension that we are slicing
                    individual_dfs[l] = torch.stack([individual_dfs[l][:,0,:,individual_dfs[l].shape[-2]//2,:], individual_dfs[l][:,self.ndims-1,:,individual_dfs[l].shape[-2]//2,:]],dim=1)
                    final_dfs[l] = torch.stack([final_dfs[l][:,0,:,final_dfs[l].shape[-2]//2,:], final_dfs[l][:,self.ndims-1,:,final_dfs[l].shape[-2]//2,:]],dim=1)
                self._log_images_in_grid(
                    y_hat[l], f"val_levels/recon level {l}")
                self._log_images_in_grid(
                    flow_to_image(individual_dfs[l]), f"val_levels/individual_DF level {l}")
                self._log_images_in_grid(
                    flow_to_image(final_dfs[l]), f"val_levels/final_DF level {l}")

        return total_loss

    @torch.no_grad()
    def _log_images_in_grid(
        self, imgs: torch.Tensor, name: str):
        bs = imgs.shape[0]
        nrow = int(math.sqrt(bs))
        imgrid = make_grid(imgs, nrow=nrow)
        self.logger.experiment.add_image(name, imgrid, self.trainer.global_step)  # type: ignore[union-attr]


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer



class PHIReg(AbstractHierarchicalProbabilisticRegistrationModel):

    """
    The actual PHIReg is now just defined by the network architecture and
    the losses
    """

    def __init__(
        self,
        total_levels: int,
        latent_levels: int,
        zdim: int,
        beta: float,
        input_size: list[int],    
        lr: float=1e-4,
        recon_loss: list=["mse"],
        ncc_factor: int=100,
        dice_factor: int=50,
        similarity_pyramid: bool=False,
        lamb: float=0,
        regularizer: str="jdet",
        image_logging_frequency: int=1000,
        decoder: str="complex",
        feedback: list=["combined_df"],
        df_resolution: str="full_res",
        df_combination: str="add",
        n0: int=32,
        segs: bool=False,
        lms: bool=False,
        mask: bool=False,
        nondiagonal: bool=False,
        cp_depth: int=3,
    ) -> None:
        super().__init__()

        from src.components.phireg import DownPath, Autoencoder, PHIRegEncoder, BSplineDecoder, SVFDecoder, PHIRegPrior

        # Make all arguments to init accessible via self.hparams (e.g.
        # self.hparams.total_levels) and save hyperparameters to checkpoint
        # and potentially logger
        self.save_hyperparameters()

        sampler: SamplerType = gauss_sampler

        torch.autograd.set_detect_anomaly(True)

        # Could access these vars via self.hparams but make member of self for less clutter
        self.segs = segs
        self.lms = lms
        self.mask = mask
        self.latent_levels = latent_levels
        self.total_levels = total_levels
        self.lk_offset = total_levels - latent_levels
        self.beta = beta
        self.df_resolution = df_resolution
        self.df_combination = df_combination
        self.recon_loss = recon_loss
        self.input_size = input_size
        self.ndims = len(input_size)
        self.cp_depth = cp_depth
        print("INPUT SIZE: ", self.input_size)
        # latent level sizes
        self.level_sizes = {key: torch.tensor(self.input_size) // (2**(key+self.lk_offset)) for key in range(self.latent_levels)}
        print("LEVEL SIZES: ", self.level_sizes)

        if self.df_combination == "add":
            self.df_combiner = DFAdder()        

        self.prior = PHIRegPrior()
        self.downpath = DownPath(
            total_levels = self.total_levels,
            latent_levels = self.latent_levels,
            input_size = self.input_size,
            input_channels = 2,
            n0 = n0,
            )
        self.autoencoder = Autoencoder(
            sampler = sampler,
            decoder = self.hparams.decoder,
            total_levels = self.total_levels,
            latent_levels = self.latent_levels,
            zdim = len(self.input_size),
            input_size = self.input_size,
            feedback = self.hparams.feedback,
            df_resolution = self.hparams.df_resolution,
            n0 = n0,
            cp_depth=self.cp_depth,
            )

        if self.hparams.regularizer == "jdet":
            regularization_loss=JDetStd
        elif self.hparams.regularizer == "L2":
            regularization_loss=L2_reg
        else:
            raise ValueError(f"Hyperparameter regularizer is {self.hparams.regularizer}. Not a known option.")
        print("REGULARIZER: ", self.hparams.regularizer)            

        # window size for the NCC loss
        window_size = {l: 1+2*(self.latent_levels-l) for l in range(self.latent_levels)}
        if self.latent_levels == 1:
            window_size = {0: 9}
        print("WINDOW SIZE: ", window_size)
        
        # these dictionaries scale the losses so that lower levels have the same loss magnitude as the high levels
        scale_dict = {l: (2.0**self.ndims)**l for l in range(latent_levels)}
        kl_weight_dict = scale_dict.copy()

        if self.df_resolution == "full_res":
            recon_weight_dict = {l: 1.0 for l in range(latent_levels)}
            regularization_weight_dict = {l: 1.0 for l in range(latent_levels)}
        else:
            # the highest level has loss calculated at full resolution, so we need to reduce its impact based on lk_offset
            recon_weight_dict = scale_dict.copy()
            recon_weight_dict[0] = scale_dict[0] / (2**(self.ndims*self.lk_offset))
            regularization_weight_dict = scale_dict.copy()
            regularization_weight_dict[0] = scale_dict[0] / (2**(self.ndims*self.lk_offset))
            # recon_weight_dict = {l: scale_dict[0] if l==0 else scale_dict[l+self.lk_offset-1] for l in range(latent_levels)}
            # regularization_weight_dict = {l: scale_dict[0] if l==0 else scale_dict[l+self.lk_offset-1] for l in range(latent_levels)}
        # HACK: Keine Ahnung warum das nötig ist
        recon_weight_dict[0] *= 4
        print("KL WEIGHT DICT: ", kl_weight_dict)
        print("RECON WEIGHT DICT: ", recon_weight_dict)
        print("REGULARIZATION WEIGHT DICT: ",regularization_weight_dict)


        KL_loss = KL_nondiagonal if nondiagonal else KL_two_gauss_with_diag_cov
        self.hierarchical_kl_loss = HierarchicalKLLoss(
            KL_divergence=KL_loss, weight_dict=kl_weight_dict, similarity_pyramid=self.hparams.similarity_pyramid, level_sizes=self.level_sizes
        )
        self.hierarchical_recon_loss = HierarchicalReconstructionLoss(
            recon_loss=self.recon_loss, weight_dict=recon_weight_dict, similarity_pyramid=self.hparams.similarity_pyramid, window_size=window_size, ndims=self.ndims
        )
        self.hierarchical_regularization = HierarchicalRegularization(
            regularizer=regularization_loss, weight_dict=regularization_weight_dict, similarity_pyramid=self.hparams.similarity_pyramid
        )