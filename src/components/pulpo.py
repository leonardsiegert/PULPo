import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from src.custom_types import SamplerType, OutputDictType
from src.utils import ModuleIntDict
from src.network_blocks import ConvSequence, MuSigmaBlock, ControlPoints, SpatialTransformer, ResizeTransform, DFAdder, BSplineInterpolate, VecInt


# Note about indexing in this class: PULPo has two types of levels:
# latent levels and resolution (total) levels. In this class latent
# levels are always index by l, total levels are indexed by k. All
# layers are indexed using total levels (k). All outputs in the
# forward pass are indexed using latent levels (l).
class DownPath(nn.Module):
    def __init__(
        self,
        total_levels: int,
        latent_levels: int,
        input_size: list[int],
        input_channels: int = 2,
        n0: int = 32,
    ) -> None:

        super().__init__()
        self.total_levels = total_levels
        self.latent_levels = latent_levels
        self.lk_offset = total_levels - latent_levels
        self.input_size = input_size

        # increase num_channels until the 4th level, then use n0*6 channels
        num_channels = {
            k: n0 * v for k, v in enumerate([1, 2, 4] + [6] * (total_levels - 3))
        }

        # Create upsampling and downsampling layers (those can be reused)
        ndims = len(self.input_size)
        AvgPool = getattr(nn, 'AvgPool%dd' % ndims)
        self.downsample = AvgPool(
            kernel_size=2, stride=2, padding=0, ceil_mode=True
        )
        
        # Create layers of main downstream path
        self.down_blocks = ModuleIntDict(
            {
                0: ConvSequence(
                    input_size=self.input_size,in_channels=input_channels, out_channels=num_channels[0], depth=3
                )
            }
        )
        for k in range(1, total_levels):
            self.down_blocks[k] = ConvSequence(
                input_size=self.input_size,
                in_channels=num_channels[k - 1],
                out_channels=num_channels[k],
                depth=3,
            )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
    ) -> OutputDictType:
        
        # concatenate the two images
        x = torch.cat([x, y], dim=1)

        # Going all the way down on the encoding path
        down_activations = {0: self.down_blocks[0](x)}
        for k in range(1, self.total_levels):
            down_sampled = self.downsample(down_activations[k - 1])
            down_activations[k] = self.down_blocks[k](down_sampled)
        # print all down activations shapes
        # for k in range(self.total_levels):
        #     print(f"down_activations[{k}].shape: {down_activations[k].shape}")

        return down_activations


class Autoencoder(nn.Module):
    """Leveled Autoencoder that performs encoding and decoding on all latent levels."""

    def __init__(
        self,
        sampler: SamplerType,
        decoder: str,
        total_levels: int,
        latent_levels: int,
        zdim: int,
        input_size: list[int],
        feedback: list[str],
        df_resolution: str,
        n0: int = 32,
        cp_depth: int = 3,
    ) -> None:

        super().__init__()
        self.sampler = sampler
        self.total_levels = total_levels
        self.latent_levels = latent_levels
        self.lk_offset = total_levels - latent_levels
        self.input_size = input_size
        self.feedback = feedback
        self.df_resolution = df_resolution
        self.cp_depth = cp_depth

        # dictionary of the sizes of the downactivations and thus also the encoder samples z
        self.level_sizes = {0:[dim for dim in input_size]}
        for k in range(total_levels-1):
            curr = torch.ceil(torch.tensor(self.level_sizes[k])/2)
            self.level_sizes[k+1] = [int(curr[i].item()) for i in range(len(input_size))]
            
        # increase num_channels until the 4th level, then use n0*6 channels
        num_channels = {
            k: n0 * v for k, v in enumerate([1, 2, 4] + [6] * (total_levels - 3))
        }

        # Create layers for feeding back latent variables to the hierarchy level above
        up_block_channels = 0
        for item in self.feedback:
                    if item == "samples":
                        up_block_channels += zdim
                    elif item == "transformed":
                        up_block_channels += 1
                    elif item in ["control_points", "individual_dfs","combined_dfs", "final_dfs"]:
                        up_block_channels += len(input_size)
                    else:
                        raise ValueError(f"Feedback list contains {item}. Not a known option.")
        self.up_blocks = ModuleIntDict()
        for k in range(total_levels - latent_levels, total_levels - 1):
            self.up_blocks[k] = ConvSequence(
                input_size=self.input_size,
                in_channels=up_block_channels,
                out_channels=n0 * zdim,
                depth=2,
            )

        self.encoders = ModuleIntDict()
        for l in range(latent_levels):
            self.encoders[l] = PULPoEncoder(
                sampler = self.sampler,
                num_channels = num_channels[self.lk_offset + l],
                zdim = zdim,
                input_size = self.level_sizes[self.lk_offset + l],
                n0 = n0,
            )

        
        self.decoders = ModuleIntDict()
        if decoder == "SVF":
            self.decoder = SVFDecoder
        else:
            raise ValueError(f"Decoder is {decoder}. Not a known option.")
        for l in range(latent_levels):
            # print(f"On latent level {l} insize is {self.level_sizes[self.lk_offset + l]} and outsize is {self.level_sizes[self.lk_offset + l]}.")
            self.decoders[l] = self.decoder(        
                zdim = zdim,
                # TODO: This insize outsize logic only applies to B-splines
                # for SVFs, the insize is the outsize
                # instead of the insize, we should have a parameter that determines the size of the previous level's control point grid
                # maybe we can just use the insize of the previous level, and the outsize of the current level
                insize = self.level_sizes[self.lk_offset + l],
                # if we have full res outputs, input size equals output size
                # on the highest level, we also always want full res outputs
                # HACK: hardcoded this, change it !!!
                # outsize = self.input_size if (self.df_resolution == "full_res" or l==0) else self.level_sizes[self.lk_offset + l - 1],
                outsize = self.input_size if (self.df_resolution == "full_res" or l==0) else self.level_sizes[self.lk_offset + l],
                df_resolution = self.df_resolution,
                n0 = n0,
                cp_depth=self.cp_depth,
            )
        
        ndims = len(input_size)
        self.avgPool = getattr(F, 'avg_pool%dd' % ndims)
        if ndims == 2:
            self.mode = 'bilinear'
        else:
            self.mode = 'trilinear'


    def forward(
        self,
        x: torch.Tensor,
        down_activations: ModuleIntDict,
        deterministic: bool = False,
    ) -> tuple[OutputDictType, OutputDictType, OutputDictType, OutputDictType, OutputDictType, OutputDictType, OutputDictType]:

        # dictionary for x on the scale of the latent levels. highest level having original size
        if self.df_resolution == "full_res":
            level_x = {l: x for l in range(self.latent_levels)}
        else:
            level_x = {0:x}
            # bring it to the size of the first latent level
            # HACK: hardcoded this, change it !!!
            for k in range(self.lk_offset):
            # for k in range(self.lk_offset-1):
                level_x[0] = self.avgPool(level_x[0], kernel_size=2, stride=2, padding=0, ceil_mode=True)
            # bring it to the size of all latent levels
            for l in range(1, self.latent_levels):
                level_x[l] = self.avgPool(level_x[l-1], kernel_size=2, stride=2, padding=0, ceil_mode=True)
            # set the original image on the lowest level
            level_x[0] = x

        # # print all level_x keys and shapes
        # for l in range(self.latent_levels):
        #     print(f"level_x[{l}].shape: {level_x[l].shape}")

        # Going back up (switching to indexing by latent level)
        mus, sigmas, samples, control_points, individual_dfs, combined_dfs, final_dfs, transformed = {}, {}, {}, {}, {}, {}, {}, {}
        for l in reversed(range(self.latent_levels)):
            k = l + self.lk_offset
            # on the lowest level
            if l == self.latent_levels - 1:
                mus[l], sigmas[l], samples[l] = self.encoders[l](down_activations[k])
                if deterministic:
                    control_points[l], individual_dfs[l], combined_dfs[l], final_dfs[l], transformed[l] = self.decoders[l](mus[l], level_x[l])
                else:
                    control_points[l], individual_dfs[l], combined_dfs[l], final_dfs[l], transformed[l] = self.decoders[l](samples[l], level_x[l])
            # on all other levels
            else:
                # the feedback connection concatenates the variables given in the parser
                feedback = []
                down_size = down_activations[k].size()[2:]
                for item in self.feedback:
                    try:
                        feedback.append(F.interpolate(locals()[item][l+1],size=down_size,mode=self.mode, align_corners=False))  
                    except:
                        raise ValueError(f"Feedback list contains {item}. Not a known option.")

                sample_upsampled = torch.cat(feedback, dim=1)
                ic(sample_upsampled.shape)
                sample_upsampled = self.up_blocks[k](sample_upsampled)
                # print("feedback shape: ", sample_upsampled.shape)

                mus[l], sigmas[l], samples[l] = self.encoders[l](down_activations[k], feedback=sample_upsampled)
                if deterministic:
                    control_points[l], individual_dfs[l], combined_dfs[l], final_dfs[l], transformed[l] = self.decoders[l](mus[l], level_x[l], combined_df=combined_dfs[l+1])
                else:    
                    control_points[l], individual_dfs[l], combined_dfs[l], final_dfs[l], transformed[l] = self.decoders[l](samples[l], level_x[l], combined_df=combined_dfs[l+1])
        return mus, sigmas, samples, control_points, individual_dfs, combined_dfs, final_dfs, transformed



class PULPoEncoder(nn.Module):
    def __init__(
        self,
        sampler: SamplerType,
        num_channels: int,
        zdim: int,
        input_size: list[int],
        n0: int = 32,
    ) -> None:
        
        super().__init__()
        self.sampler = sampler
        self.num_channels = num_channels
        self.zdim = zdim

        self.sample_merge_block = ConvSequence(
                    input_size=input_size,
                    in_channels=num_channels + n0 * zdim,
                    out_channels=num_channels,
                    depth=2,
                )
        
        self.mu_sigma = MuSigmaBlock(input_size=input_size, in_channels=num_channels, zdim=zdim)


    def forward(
        self,
        down_activation: torch.Tensor,
        feedback: Optional[torch.Tensor] = None,
    ) -> tuple[OutputDictType, OutputDictType, OutputDictType]:

        # on the lowest level
        if feedback is None:
            mu, sigma = self.mu_sigma(down_activation)
        # on all other levels
        else:
            ic(feedback.shape, down_activation.shape)
            intermediate = torch.cat([feedback, down_activation], dim=1)
            intermediate = self.sample_merge_block(intermediate)
            # print("shape before sampling: ", intermediate.shape)
            mu, sigma = self.mu_sigma(intermediate)

        # Generate samples
        z = self.sampler(mu, sigma)

        return mu, sigma, z

class SVFDecoder(nn.Module):
    def __init__(
        self,
        zdim: int,
        insize: list[int],
        outsize: list[int],
        df_resolution: str,
        n0: int = 32,
        cp_depth: int = 3,
    ) -> None:
        super().__init__()
        self.zdim = zdim
        self.insize = insize
        self.outsize = outsize
        self.cp_depth = cp_depth
        
        # turning the sample z into control points
        self.control_points = ControlPoints(input_size=self.insize, zdim=self.zdim, max_channels=n0, depth=self.cp_depth)

        # with svfs, this control point grid is the deformation field
        
        # combines the DFs
        # Hardcoding this because why would it ever be different?
        # THIS would not be different on full res:
        self.vel_resize_feedback = 1 / 2 # this should be 0.5 for everything EVEN on the highest level, because we compute full res prediction after integration (?!)
        self.resizer_feedback = ResizeTransform(self.vel_resize_feedback, ndims=len(self.insize))
        # THIS would be different on full res:
        self.vel_resize_output = 1 / (self.outsize[0] / self.insize[0]) # bc with SVFs we also have DFs the size of the CP grid, and then resize everything to output size
        self.resizer_output = ResizeTransform(self.vel_resize_output, ndims=len(self.insize))
        
        self.combined_deformation_field = DFAdder()

        # perform vector field integration to make the vector field diffeomorphic
        self.integrate = VecInt(self.insize,nsteps=7)
        
        
        self.spatial_transform = SpatialTransformer(self.outsize)


    # TODO: check if the tranformations leave the initial thing intact
    # i.e. is the combined_df the same after integration?
    def forward(self, z: torch.Tensor, input_image: torch.Tensor, combined_df: Optional[torch.Tensor]=None) -> tuple[OutputDictType, OutputDictType, OutputDictType, OutputDictType]:
        if combined_df != None:
            ic(combined_df.shape)
        # turning the sample z into control points
        individual_df = self.control_points(z)
        ic(individual_df.shape)
        # combine the DFs
        if combined_df is None: # on the lowest level
            combined_df = individual_df
        else:
            combined_df = self.combined_deformation_field(self.resizer_feedback(combined_df), individual_df)
        ic(combined_df.shape)
        

        # # resize the combined df to the output siz
        # combined_df = self.resizer_output(combined_df)
        # ic(combined_df.shape)
        # # perform vector field integration to make the vector field diffeomorphic
        # integrated_df = self.integrate(combined_df)
        # ic(integrated_df.shape)


        # perform vector field integration to make the vector field diffeomorphic
        integrated_df = self.integrate(combined_df)
        ic(integrated_df.shape)

        # resize the integrated df to the output size
        integrated_df = self.resizer_output(integrated_df)
        ic("final size: ",integrated_df.shape)
        ic("input_image size: ",input_image.shape)
        
        # spatially transform the image
        transformed_image = self.spatial_transform(integrated_df, input_image)

        return individual_df, individual_df, combined_df, integrated_df, transformed_image



class PULPoPrior(nn.Module):
    def __init__(
        self,
    ) -> None:
        super().__init__()

    def forward(
        self,
        posterior_mus:OutputDictType,
        posterior_sigmas:OutputDictType,
    ) -> tuple[OutputDictType, OutputDictType]:
        prior_mus, prior_sigmas = {}, {}

        for l in posterior_mus.keys():
            prior_mus[l] = torch.zeros_like(posterior_mus[l], dtype=torch.float32)
            prior_sigmas[l] = torch.ones_like(posterior_sigmas[l], dtype=torch.float32)

        return prior_mus, prior_sigmas