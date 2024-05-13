# necessary imports
import os
import glob
import argparse
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import cm
from matplotlib.pyplot import get_cmap 
import seaborn as sns
import numpy as np
import pandas as pd
import torch
import scipy.interpolate as si

import gc
from torchvision.utils import flow_to_image
from torchvision.transforms.functional import gaussian_blur
from src.losses import HierarchicalReconstructionLoss, HierarchicalRegularization, L2_loss, Soft_dice_loss, NCC_loss, jacobian_det,JDetStd
from src.data.BraTS import brats
from src.data.OASIS import oasis
from src.models import PULPo

os.environ['NEURITE_BACKEND'] = 'pytorch'
# TODO: import vxm?
os.environ['VXM_BACKEND'] = 'pytorch'

# seeding for reproducibility
torch.manual_seed(0)
np.random.seed(0)


class Evaluate():
    def __init__(self):
        self.checkpoint_folder = "checkpoints/best-reconstruction*.ckpt"
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        #####################################################################################################
        ####### LOAD DATA               #####################################################################
        #####################################################################################################
        self.num_inputs = ...
        self.loaders = ...
        self.loader_names = ...
        self.segs = False
        self.lms = False
        self.mask = False
        self.grid_size = 20

        #####################################################################################################
        ####### SELECT METRICS               ################################################################
        #####################################################################################################
        self.metric_names = ...
        self.metrics = ...
        self.num_datasets, self.num_metrics = (..., ...)

        self.model = None
        self.latent_levels = ...
        self.models = ...
        self.num_models = ...
        self.model_names = ...
        self.output_dir = ...

    #####################################################################################################
    ####### HELPERS FOR DATA LOADING AND MODEL PASSES    ################################################
    #####################################################################################################

    def build_path(self, model_dir,name):
        """ Builds the path to the checkpoint file."""
        filepath = model_dir+"/"+name+"/"+self.checkpoint_folder
        checkpoint = glob.glob(filepath)[0]
        return checkpoint
    
    def load_model(self, model_dir, git_hash, version):
        """ Loads the PULPo model from a given directory. git_has and version required as this is 
            how the trained models are saved."""
        name = git_hash+"/"+version
        checkpoint = self.build_path(model_dir, name)
        self.output_dir = model_dir + "/" + name + "/" + "evaluation"
        os.makedirs(self.output_dir, exist_ok=True)
        model = PULPo.load_from_checkpoint(checkpoint).to(self.device)
        model.eval()
        self.model = model
        self.latent_levels = model.latent_levels
        return model
    
    def load_vxm(self,model_dir,name):
        """ Loads the DIF-VM (vxm) baseline model from a given directory."""
        path = model_dir+"/"+name+".pt"
        self.model = vxm.networks.VxmDense.load(path, self.device)
        self.model.to(self.device)
        return self.model
    
    def load_data(self, task, segs, lms, mask, ndims):
        """ Loads the data loaders for the given task."""
        self.segs = segs
        self.lms = lms
        self.mask = mask

        if task == "oasis":
            self.task = "oasis"
            (
                self.train_loader,
                self.validation_loader,
                self.test_loader_seg,
                self.test_loader_lm,
            ) = oasis.create_data_loaders(batch_size=1, segs=segs, lms=lms, mask=mask, ndims=ndims)
            self.loaders = [self.train_loader,self.validation_loader,self.test_loader_seg,self.test_loader_lm]
            self.loader_names = ["train","val","test_seg","test_lm"]
        elif task == "brats":
            self.task = "brats"
            (
            self.train_loader,
            self.validation_loader,
            self.test_loader,
            ) = brats.create_data_loaders(batch_size=1, segs=segs, lms=lms, mask=mask, ndims=ndims)
            self.loaders = [self.train_loader,self.validation_loader,self.test_loader]
            self.loader_names = ["train","val","test"]
        else:
            raise Exception(f"Task {task} does not exist." )
        
        # metrics
        self.metrics = [L2_loss, JDetStd, jacobian_det]
        self.metric_names = ["RMSE", "JDetStd", "JDetLeq0"]
        if self.segs:
            self.metrics += [Soft_dice_loss]
            self.metric_names += ["Dice"]
        if self.lms:
            self.metrics += [self.lm_mae, self.lm_euclid]
            self.metric_names += ["LM_MAE", "LM_Euclid"]
        self.num_datasets, self.num_metrics = len(self.loaders), len(self.metrics)
        self.num_inputs = self.train_loader.dataset.__len__()
        return
    
    def sample_data(self, loader_name, index=0):
        """ Samples one datapoint from the given loader. """
        # check which element in self.loader_names corresponds to loader_name
        loader_idx = self.loader_names.index(loader_name)
        loader = self.loaders[loader_idx]

        if index >= len(loader):
            raise ValueError(f"Index {index} is out of range for loader {loader_name}.")
        if index != 0:
            print(f"Warning: You are not using the first element of the loader {loader_name}.")
            for i, input in enumerate(loader):
                if i == index:
                    break
        else:
            input = next(iter(loader))
        x,y,seg1,seg2,lm1,lm2,mask_x,mask_y = input
        return [x.to(self.device), y.to(self.device), seg1.to(self.device), seg2.to(self.device), lm1.to(self.device), lm2.to(self.device), mask_x.to(self.device), mask_y.to(self.device), loader_name]

    def predict(self, inputs, num_samples=20, deterministic=False):
        """ Generates a prediction given the inputs with the current model."""
        with torch.no_grad():
            model = self.model
            x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs

            if deterministic and num_samples != 1:
                raise Exception("Deterministic predictions with more than 1 sample make no sense!")

            if num_samples==1:
                if deterministic:
                    outputs, individual_dfs = model.predict_deterministic(x,y)
                    prediction_name = "deterministic_prediction"
                else:
                    outputs, individual_dfs = model.predict(x,y, N=num_samples)
                    prediction_name = "sample_prediction"

                combined_dfs, final_dfs = model.combine_dfs(individual_dfs)
                if self.segs:
                    warped_seg = {key:model.autoencoder.decoders[key].spatial_transform(final_dfs[key], seg_x) for key in final_dfs}
                else:
                    warped_seg = {}
                    warped_seg[0] = torch.empty((0,), dtype=torch.float32)

                return ([outputs[0], final_dfs[0], warped_seg[0], outputs, individual_dfs, combined_dfs,final_dfs, warped_seg, prediction_name],[])
                   
            else: # num_samples > 1
                prediction_name = f"avg_prediction_over_{str(num_samples)}_samples"
                if self.model.ndims == 3:
                    # we don't use segmentations for 3D uncertainty due to memory constraints
                    warped_seg = {}
                    warped_seg[0] = torch.empty((0,), dtype=torch.float32)
                    all_warped_seg = warped_seg
                    
                    # we can't batch 3D predictions, so we have to predict iteratively
                    latent_levels = model.latent_levels
                    level_sizes = {0:[dim for dim in model.input_size]}
                    for k in range(model.total_levels-1):
                        curr = torch.ceil(torch.tensor(level_sizes[k])/2)
                        level_sizes[k+1] = [int(curr[i].item()) for i in range(len(model.input_size))]
                    # the outputs and final df are on the highest resolution on level 0
                    all_outputs = {key: torch.zeros((num_samples,1,*level_sizes[0]),device=self.device) if key==0 else
                                torch.zeros((num_samples,1,*level_sizes[key+model.lk_offset]),device=self.device) for key in range(latent_levels)}
                    all_final_dfs = {key: torch.zeros((num_samples,3,*level_sizes[0]),device=self.device) if key==0 else
                                    torch.zeros((num_samples,3,*level_sizes[key+model.lk_offset]),device=self.device) for key in range(latent_levels)}
                    # the individual and combined df are not necessarily on the highest resolution on level 0
                    all_individual_dfs = {key: torch.zeros((num_samples,3,*level_sizes[key+model.lk_offset]),device=self.device) for key in range(latent_levels)}
                    all_combined_dfs = {key: torch.zeros((num_samples,3,*level_sizes[key+model.lk_offset]),device=self.device) for key in range(latent_levels)}
                    
                    for i in range(num_samples):
                        outputs, individual_dfs = model.predict(x,y, N=1)
                        combined_dfs, final_dfs = model.combine_dfs(individual_dfs)
                        for key in range(latent_levels):
                            all_outputs[key][i] = outputs[key][0]
                            all_individual_dfs[key][i] = individual_dfs[key][0]
                            all_combined_dfs[key][i] = combined_dfs[key][0]
                            all_final_dfs[key][i] = final_dfs[key][0]
                    
                    # calculate the averages
                    individual_dfs = {key:individual_dfs[key].mean(dim=0).unsqueeze(0) for key in all_individual_dfs}
                    combined_dfs, final_dfs = model.combine_dfs(individual_dfs)
                    outputs = {key:model.autoencoder.decoders[key].spatial_transform(final_dfs[key], x) for key in final_dfs}

                    # calculate the stds
                    output_std = {key:torch.mean(torch.std(all_outputs[key], axis=0),axis=0) for key in all_outputs}
                    if self.mask:
                        # mask the dfs for std calculation. The mask is first pooled to the level sizes and then warped.
                        warped_mask = {key:model.autoencoder.decoders[key].spatial_transform(final_dfs[key], mask_x) for key in final_dfs}
                        individual_df_std = {key:torch.mean(torch.std(all_individual_dfs[key], axis=0),axis=0) for key in all_outputs}
                        final_df_std = {key:torch.mean(torch.std(all_final_dfs[key]*model.autoencoder.decoders[key].spatial_transform(final_dfs[key], mask_x), axis=0),axis=0) for key in all_outputs}
                    else:
                        individual_df_std = {key:torch.mean(torch.std(all_individual_dfs[key], axis=0),axis=0) for key in all_outputs}
                        final_df_std = {key:torch.mean(torch.std(all_final_dfs[key], axis=0),axis=0) for key in all_outputs}
                
                else: # ndims == 2
                    outputs, individual_dfs = model.predict_output_samples(x,y, N=num_samples)
                    if self.segs:    
                        seg_xn = torch.vstack([seg_x for _ in range(num_samples)])
                    else:
                        warped_seg = {}
                        warped_seg[0] = torch.empty((0,), dtype=torch.float32)
                        all_warped_seg = warped_seg

                    # save these for later
                    all_outputs = {key:outputs[key].squeeze(0) for key in outputs}
                    all_individual_dfs = {key:individual_dfs[key].squeeze(0) for key in individual_dfs}

                    individual_dfs = {key:individual_dfs[key].mean(dim=1) for key in individual_dfs}
                    combined_dfs, final_dfs = model.combine_dfs(individual_dfs)
                    all_combined_dfs, all_final_dfs = model.combine_dfs(all_individual_dfs)
                    if self.segs:
                        warped_seg = {key:model.autoencoder.decoders[key].spatial_transform(final_dfs[key], seg_x) for key in final_dfs}
                        all_warped_seg = {key:model.autoencoder.decoders[key].spatial_transform(all_final_dfs[key], seg_xn) for key in all_final_dfs}

                    outputs = {key:model.autoencoder.decoders[key].spatial_transform(final_dfs[key], x) for key in final_dfs}
                    # calculate the stds
                    output_std = {key:torch.mean(torch.std(all_outputs[key], axis=0),axis=0) for key in all_outputs}
                    individual_df_std = {key:torch.mean(torch.std(all_individual_dfs[key], axis=0),axis=0) for key in all_outputs}
                    final_df_std = {key:torch.mean(torch.std(all_final_dfs[key], axis=0),axis=0) for key in all_outputs}
                    
                return ([outputs[0], final_dfs[0], warped_seg[0], outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name],
                        [output_std, individual_df_std, final_df_std, all_outputs, all_individual_dfs, all_combined_dfs, all_final_dfs, all_warped_seg])

    def predict_vxm(self, moving, fixed, num_samples=20, deterministic=False):
        """ Generates a prediction given the inputs with the current DIF-VM baseline model."""
        if deterministic and num_samples != 1:
            raise Exception("Deterministic predictions can only be made for 1 sample.")
        if deterministic and num_samples == 1:
            prediction_name = "deterministic_prediction"
        else:
            prediction_name = f"avg_prediction_over_{str(num_samples)}_samples"

        with torch.no_grad():
            inshape = moving.shape[2:]
            all_moved = torch.zeros((num_samples,1,*inshape), device=moving.device)
            all_warp = torch.zeros((num_samples,3,*inshape), device=moving.device)

            for i in range(num_samples):
                moved, warp, _ = self.model(moving, fixed, registration=True, deterministic=deterministic)
                all_moved[i] = moved[0]
                all_warp[i] = warp[0]
            # calculate the averages
            avg_moved = torch.mean(all_moved, axis=0).unsqueeze(0)
            avg_warp = torch.mean(all_warp, axis=0).unsqueeze(0)
            # calculate the stds
            moved_std = torch.mean(torch.std(all_moved, axis=0),axis=0)
            warp_std = torch.mean(torch.std(all_warp, axis=0),axis=0)

        
        return ([avg_moved, avg_warp, [], [], [], [], [], [], prediction_name],
                        [moved_std, [], warp_std, all_moved, [], [], all_warp, []])

    ##################################################################################################################
    ########  METRICS                                         ########################################################
    ##################################################################################################################

    def rmse(self, input,target):
        """ Computes the root mean squared error between two images. """
        criterion = torch.nn.MSELoss()
        mse = criterion(input, target)
        return torch.sqrt(mse)

    def dsc(self, input,target):
        """ Computes the dice similarity coefficient between two segmentation maps. """
        input_size = input.size()[2:]
        sumdims = list(range(2, len(input_size) + 2))
        epsilon = 1e-6
        dsc = ((2. * target*input).mean(dim=sumdims) + epsilon) / ((target**2).mean(dim=sumdims) + (input**2).mean(dim=sumdims) + epsilon)
        return dsc.mean()

    def jdet(self, df):
        """ Computes the jacobian determinant of a displacement field. """
        jdet = jacobian_det(df, normalize=True)
        return jdet

    def ncc(self,a,v, zero_norm=True):
        """Computes the normalized cross correaltion between two arrays

        Args:
            a (np.array): first array
            v (np.array): second array

        Returns:
            float: the normalized cross correlation between arrays a and v
        """
        a = a.flatten()
        v = v.flatten()
        eps = 1e-15
        if zero_norm:
            a = (a - np.mean(a)) / (np.std(a) * len(a) + eps)
            v = (v - np.mean(v)) / (np.std(v) + eps)
        else:
            a = (a) / (np.std(a) * len(a) + eps)
            v = (v) / (np.std(v) + eps)
        return np.correlate(a, v)[0]

    def lm_mae(self, lm1, lm2):
        """ Computes the median manhattan distance (median absolute error) between two sets of landmarks
            Args:
                lm1 (torch.Tensor): first set of landmarks
                            Shape: (1, n_landmarks, 3)
                lm2 (torch.Tensor): second set of landmarks
                            Shape: (1, n_landmarks, 3)
            Returns:
                float: the median manhattan distance between the landmarks
        """
        distance = torch.abs(lm1-lm2).sum(dim=2)
        return torch.median(distance)
    
    def lm_euclid(self, lm1, lm2):
        """ Computes the mean euclidean distance between two sets of landmarks
            Args:
                lm1 (torch.Tensor): first set of landmarks
                            Shape: (1, n_landmarks, 3)
                lm2 (torch.Tensor): second set of landmarks
                            Shape: (1, n_landmarks, 3)
            Returns:
                float: the mean euclidean distance between the landmarks
        """
        distance = torch.sqrt(((lm1-lm2)**2).sum(dim=2))
        return torch.mean(distance)

    def lms_var(self, lms):
        """ Computes the variance of the euclidean distance between landmarks
            Args:
                lms (torch.Tensor): set of landmarks
                            Shape: (n_samples, n_landmarks, 3)
            Returns:
                torch.Tensor: the variance of the euclidean distance between landmarks
                    Shape: (n_landmarks,3)
        """
        return torch.mean(torch.var(lms, dim=0),dim=-1)
    
    def lms_corr(self, lm_hat, lms, lm):
        """ Computes the normalized cross correlation between the mean squared error and the variance of the landmarks
            Args:
                lm_hat (torch.Tensor): predicted landmarks
                            Shape: (n_landmarks, 3)
                lms (torch.Tensor): set of sample prediction landmarks
                            Shape: (n_samples, n_landmarks, 3)
                lm (torch.Tensor): ground truth landmarks
                            Shape: (n_landmarks, 3)
            Returns:
                float: the normalized cross correlation between the mean squared error and the variance of the landmarks
        """
        error = torch.mean((lm_hat - lm)**2, dim=-1).flatten()
        variance = self.lms_var(lms).flatten()
        error_normed = (error - torch.mean(error)) / (torch.std(error) * len(error))
        variance_normed = (variance - torch.mean(variance)) / (torch.std(variance))
        return np.correlate(error_normed.to("cpu"), variance_normed.to("cpu"))[0]

    def warp_landmarks(self, lm: torch.Tensor, df:torch.Tensor) -> torch.Tensor:
        """ Warps a set of landmarks using a displacement field
            Args:
                lm (torch.Tensor): set of landmarks
                            Shape: (1, n_landmarks, 3)
                df (torch.Tensor): displacement field
                            Shape: (n_samples, ndims, H, W, D)
            Returns:
                torch.Tensor: the warped landmarks
                    Shape: (1, n_landmarks, 3)
        """
        lm = lm.long()
        new_lm = lm - df[:,:,lm[0,:,0],lm[0,:,1],lm[0,:,2]].transpose(-2,-1)
        return new_lm

    ##################################################################################################################
    ########  Helpers for tables and visualizations           ########################################################
    ##################################################################################################################

    # adapted from https://stackoverflow.com/questions/24612626/b-spline-interpolation-with-python
    def bspline_interpolate(self, points):
        """ Interpolates a set of points using a b-spline. We use this to interpolate the 
            line segments of the control point grid that visualizes the displacement fields.
        """
        points = points.tolist()
        degree = 3

        dist_x_left = points[1][0]-points[0][0]
        y_left = points[0][1]
        dist_x_right = points[-1][0]-points[-2][0]
        y_right = points[-1][1]
        
        points = [[points[0][0] - dist_x_left,y_left]] + points + [[points[-1][0] + dist_x_right,y_right],[points[-1][0] + 2*dist_x_right,y_right]]
        points = np.array(points)
        n_points = len(points)
        x = points[:,0]
        y = points[:,1]

        t = range(len(x))
        ipl_t = np.linspace(1.0, len(points) - degree, 1000)

        x_tup = si.splrep(t, x, k=degree, per=1)
        y_tup = si.splrep(t, y, k=degree, per=1)
        x_list = list(x_tup)
        xl = x.tolist()
        x_list[1] = [0.0] + xl + [0.0, 0.0, 0.0, 0.0]

        y_list = list(y_tup)
        yl = y.tolist()
        y_list[1] = [0.0] + yl + [0.0, 0.0, 0.0, 0.0]

        x_i = si.splev(ipl_t, x_list)
        y_i = si.splev(ipl_t, y_list)

        return np.stack((np.array(x_i), np.array(y_i)), axis=1)
    

    def create_warped_grid(self, df, grid_size):
        """ Creates a grid of control points that are warped by the displacement field.
            This grid is used to visualize the displacement field.
            Args:
                df (torch.Tensor): displacement field
                            Shape: (1, ndims, H, W(, D))
                control_points (int): number of control points
            Returns:
                grid_i, grid_j (torch.Tensor, torch.Tensor): The unstacked grid components
                        Shape: (control_points, control_points), (control_points, control_points)
        """
        
        grid_i,grid_j = torch.meshgrid(torch.linspace(0,df.shape[-2]-1,grid_size),torch.linspace(0,df.shape[-1]-1,grid_size), indexing="ij")
        grid = torch.stack((grid_i,grid_j))
        grid = grid.type(torch.float32)
        for i in range(grid.shape[-2]):
            for j in range(grid.shape[-1]):
                gi, gj = grid[:,i,j].int()
                grid[:,i,j] -= df[0,:,gi,gj]
        grid_i = grid[0,:,:]
        grid_j = grid[1,:,:]
        return grid_i, grid_j
    
    def plot_grid(self,x,y, scatter=False, lines=True, straightlines=False, ax=None, **kwargs):
        """ Plots a grid of control points that are warped by the displacement field.
            This grid is used to visualize the displacement field.
            Args:
                x (torch.Tensor): The unstacked x-components of the grid
                            Shape: (control_points, control_points)
                y (torch.Tensor): The unstacked y-components of the grid
                            Shape: (control_points, control_points)
            Returns:
                None
        """
        ax = ax or plt.gca()
        # horizontal line segments
        segs1 = np.stack((y,x), axis=2)
        # create bspline-interpolated horizontal lines
        int_lines1 = np.zeros(shape=(segs1.shape[0],1000,2))
        for i in range(segs1.shape[0]):
            int_lines1[i] = self.bspline_interpolate(segs1[i])
        # vertical lines
        segs2 = segs1.transpose(1,0,2)
        # create bspline-interpolated vertical lines
        int_lines2 = np.zeros(shape=(segs1.shape[0],1000,2))
        for i in range(segs2.shape[0]):
            int_lines2[i] = self.bspline_interpolate(segs2[i])

        # optionally scatter the control points
        if scatter:
            ax.scatter(y,x, **kwargs, s=6)
        # plot the lines
        if straightlines:
            ax.add_collection(LineCollection(segs1, **kwargs))
            ax.add_collection(LineCollection(segs2, **kwargs))
        # plot the interpolated lines
        if lines:
            ax.add_collection(LineCollection(int_lines1, **kwargs))
            ax.add_collection(LineCollection(int_lines2, **kwargs))
        return

    ##################################################################################################################
    ########  Helpers for the tables                   ###############################################################
    ##################################################################################################################
    def convert_to_scientific(self,value):
        """ Converts a value to scientific notation if it is close to zero.
            Args:
                value (float): The value to convert
            Returns:
                str: The value in scientific notation
        """
        if abs(value) < 0.001 and abs(value) > 0.0:
            return format(value, '.2e')
        else:
            return value
        
    # creates a table from a df and saves it as a tex and svg file 
    # if no name is given, the table is not saved
    def make_tables(self, df, output_dir, show=False, name=None, fontsize=4):
        """ Creates a table from a dataframe and saves it as a tex and svg file.
            If no name is given, the table is not saved."""
        df = df.applymap(self.convert_to_scientific)
        # write latex table
        latex_table = df.style.to_latex()
        # export dataframe as svg file
        fig, ax = plt.subplots()
        fig.patch.set_visible(False)
        ax.axis('off')
        table = ax.table(cellText=df.values, colLabels=df.columns, rowLabels=df.index, loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        table.auto_set_column_width(col=list(range(len(df.columns))))
        fig.tight_layout()
        if show == True:
            print(latex_table)
            plt.show()
        if name != None:
            with open(output_dir+"/"+name+".tex","w+") as f:
                    f.writelines(latex_table)
            fig.savefig(output_dir+"/"+name+".svg")
        return

    def table_jdet(self, inputs, preds, output_dir=None, name="", show=True, save=False, fontsize=4):
        """ Creates a table with the standard deviation of the jacobian determinant and 
            the percentage of pixels with jacobian determinant <=0 for the combined and individual deformation fields.
        """
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds
        output_dir = output_dir if output_dir is not None else self.output_dir
        data = np.zeros((self.latent_levels,4))

        for key in final_dfs.keys():
            final_dfs[key] = final_dfs[key].to("cpu")
            individual_dfs[key] = individual_dfs[key].to("cpu")

        for l in reversed(range(self.latent_levels)):
            # combined DF
            jdet = jacobian_det(final_dfs[l]).detach()
            data[l,0] = jdet.std()
            jdet_leq0 = np.where(jdet <= 0, 1, 0)
            data[l,1] = (np.sum(jdet_leq0==1) / np.sum(np.ones_like(jdet_leq0)))*100
            # individual DF
            jdet = jacobian_det(individual_dfs[l]).detach()
            data[l,2] = jdet.std()
            jdet_leq0 = np.where(jdet <= 0, 1, 0)
            data[l,3] = (np.sum(jdet_leq0==1) / np.sum(np.ones_like(jdet_leq0)))*100
            
        supcol = np.repeat(["combined DF", "individual DF"],2)
        subcol = np.tile(["JDet std", f"% of pixels <= 0"],2)
        mux = pd.MultiIndex.from_arrays((supcol, subcol))
        df = pd.DataFrame(data, columns=mux).round(3)
        df.index.name = "Level"
        if save:
            self.make_tables(df, output_dir, show=show, name="jdet_"+name, fontsize=4)
        else:
            self.make_tables(df, output_dir, show=show, fontsize=4)
        return

    ##################################################################################################################
    ########  Space for the visualizations           #################################################################
    ##################################################################################################################

    def artifact(self, image_tensor, method, x, y, z=None):
        """
        Inserts an artificial artifact into a region of an image.
        Args:
            image_tensor (torch.Tensor): A PyTorch tensor representing the image.
            method (str): The method to use for creating the artifact. Options are "blur", "noise", "mean", "white", "black", and "checkerboard".
            x (tuple): The x-coordinates of the region to insert the artifact.
            y (tuple): The y-coordinates of the region to insert the artifact.
            z (tuple): The z-coordinates of the region to insert the artifact. Only required for 3D images.
            
        Returns:
            torch.Tensor: A new PyTorch tensor representing the augmented image.
        """
        inshape = image_tensor.shape[2:]

        if len(inshape) == 2 and z is not None:
            raise ValueError("Ddist must be None for 2D images")
        if len(inshape) == 3 and z is None:
            raise ValueError("Ddist must be specified for 3D images")

        # Select artifact region
        if z is None:
            roi = image_tensor[...,x[0]:x[1],y[0]:y[1]]
        else:
            roi = image_tensor[...,x[0]:x[1],y[0]:y[1],z[0]:z[1]]

        # Apply the blur operation to the region
        if method == "blur": # only works on 2D
            roi = gaussian_blur(roi, kernel_size=int(5*2)+1, sigma=5)
        elif method == "noise":
            my_mean = roi.mean()
            my_std = roi.std()
            roi = torch.normal(my_mean, my_std, size=roi.shape)
        elif method == "mean":
            roi = roi.mean()
        elif method == "white":
            roi = 1
        elif method == "black":
            roi = 0
        elif method == "checkerboard":
            distx = x[1] - x[0]
            disty = y[1] - y[0]
            if z is not None:
                distz = z[1] - z[0]
            rx = 0
            ry = 0
            rz = 0
            color = 1
            roi[:] = color
            while rx < distx/2:
                if z is None:
                    roi[...,rx:-rx,ry:-ry] = color
                else:
                    roi[...,rx:-rx,ry:-ry,rz:-rz] = color
                rx += int(distx/10)
                ry += int(disty/10)
                if z is not None:
                    rz += int(distz/10)
                color = 1-color
        else:
            raise ValueError("Method not recognized")

        if z is None:
            res = image_tensor.clone()
            res[...,x[0]:x[1],y[0]:y[1]] = roi
        else:
            res = image_tensor.clone()
            res[...,x[0]:x[1],y[0]:y[1],z[0]:z[1]] = roi

        return res

    # visualize one or several visualizations together in one plot
    # visualizations is a list of functions
    def visualize(self, inputs, preds, visualizations, all_preds=None, rowparams={}, title="", show=False, save_path=None):
        """ Visualizes one or several visualization together in one plot.
            Args:
                inputs (list): The inputs as returned by sample_data()
                preds (list): The predictions as returned by predict()
                visualizations (list): Which visualization methods to plot
                all_preds (list): The additional predictions as returned by predict()
                rowparams (dict): An optional dictionary giving additional parameters to selected rows
                title (str): The title of the plot
                show (bool): Whether to show the plot
                save_path (str): The path to save the plot
        """
        # SLICE TO 2D
        new_inputs = []
        new_preds = []
        new_all_preds = []
        if len(inputs[0].shape[2:]) == 3:
            for i in range(len(inputs)):
                if isinstance(inputs[i], dict):
                    # skip it if it is empty
                    # important for when there's no segmentation
                    if not inputs[i][0].numel():
                        new_inputs.append(inputs[i])
                    else:
                        new_dict = {}
                        for key in inputs[i]:
                            slice_index = inputs[i][key].shape[-2] // 2
                            new_dict[key] = inputs[i][key][...,slice_index,:]
                        new_inputs.append(new_dict)
                elif isinstance(inputs[i], str):
                    # for the dataset name
                    new_inputs.append(inputs[i])
                else:
                    if not inputs[i].numel():
                        new_inputs.append(inputs[i])
                    else:
                        slice_index = inputs[i].shape[-2] // 2
                        new_inputs.append(inputs[i][...,slice_index,:])

            for i in range(len(preds)):
                if isinstance(preds[i], dict):
                    # skip it if the dict is empty
                    # important for when there's no segmentation
                    if not preds[i][0].numel():
                        new_preds.append(preds[i])
                    else:
                        new_dict = {}
                        for key in preds[i]:
                            slice_index = preds[i][key].shape[-2] // 2
                            new_dict[key] = preds[i][key][...,slice_index,:]
                            if new_dict[key].shape[-3] == 3:
                                new_dict[key] = torch.stack([new_dict[key][...,0,:,:], new_dict[key][...,2,:,:]],dim=-3)
                        new_preds.append(new_dict)
                elif isinstance(preds[i], str):
                    # for the prediction name
                    new_preds.append(preds[i])
                else:
                    # skip it if the dict is empty
                    # important for when there's no segmentation
                    if not preds[i].numel():
                        new_preds.append(preds[i])
                    else: 
                        slice_index = preds[i].shape[-2] // 2
                        new_pred = preds[i][...,slice_index,:]
                        if new_pred.shape[-3] == 3:
                            new_pred = torch.stack([new_pred[...,0,:,:], new_pred[...,2,:,:]],dim=-3)
                        new_preds.append(new_pred)

            if all_preds is not None:
                for i in range(len(all_preds)):
                    if isinstance(all_preds[i], dict):
                        # skip it if the dict is empty
                        # important for when there's no segmentation
                        if not all_preds[i][0].numel():
                            new_all_preds.append(all_preds[i])
                        else:
                            new_dict = {}
                            for key in all_preds[i]:
                                slice_index = all_preds[i][key].shape[-2] // 2
                                new_dict[key] = all_preds[i][key][...,slice_index,:]
                                if len(new_dict[key].shape) > 2:
                                    if new_dict[key].shape[-3] == 3:
                                        new_dict[key] = torch.stack([new_dict[key][...,0,:,:], new_dict[key][...,2,:,:]],dim=-3)
                            new_all_preds.append(new_dict)
                    else:
                        # skip it if the dict is empty
                        # important for when there's no segmentation
                        if not all_preds[i].numel():
                            new_all_preds.append(all_preds[i])
                        else:
                            slice_index = all_preds[i].shape[-2] // 2
                            new_all_pred = all_preds[i][...,slice_index,:]
                            if len(new_all_pred.shape) > 2:
                                if new_all_pred.shape[-3] == 3:
                                    new_all_pred = torch.stack([new_all_pred[...,0,:,:], new_all_pred[...,2,:,:]],dim=-3)
                            new_all_preds.append(new_all_pred)
        
            inputs = new_inputs
            preds = new_preds
            all_preds = new_all_preds

        # transfer to cpu for plotting
        new_inputs = [{key: inputs[i][key].to("cpu") for key in inputs[i].keys()} if isinstance(inputs[i], dict) else inputs[i].to("cpu") for i in range(len(inputs)-1)]
        new_inputs.append(inputs[-1])
        new_preds = [{key: preds[i][key].to("cpu") for key in preds[i].keys()} if isinstance(preds[i], dict) else ([] if preds[i]==[] else preds[i].to("cpu")) for i in range(len(preds)-1)]
        new_preds.append(preds[-1])
        if all_preds != []:
            new_all_preds = [{key: all_preds[i][key].to("cpu") for key in all_preds[i].keys()} if isinstance(all_preds[i], dict) else ([] if preds[i]==[] else preds[i].to("cpu")) for i in range(len(all_preds))]
        inputs = new_inputs
        preds = new_preds
        all_preds = new_all_preds

        # visualize
        rows = len(visualizations)
        cols = 4
        fig, ax = plt.subplots(rows,cols)
        fig.set_figwidth(30)
        fig.set_figheight(30 * rows/self.latent_levels)
        # cax = fig.add_axes([0.95, 0.05, 0.02, 0.95]) #this locates the axis that is used for a colorbar. It is scaled 0 - 1. 
        if title == None:
            ...
        elif title != "":
            fig.suptitle(title + f". {preds[-1]} on the {inputs[-1]} set.", fontsize=16)
        else:
            fig.suptitle(f" {preds[-1]} on the {inputs[-1]} set.", fontsize=16)
        if len(visualizations) == 1: # necessary, so that calling ax[r,c] works even if there is only one row
            ax = [ax]
        for r in range(rows):
            if visualizations[r] in [self.vis_output_var_per_level, self.vis_individual_df_var_per_level, self.vis_final_df_var_per_level, self.vis_sample_preds, self.vis_sample_segpreds, self.vis_sample_dfs]:
                if r in rowparams.keys():
                    visualizations[r](ax[r], inputs, preds, all_preds, **rowparams[r])
                else:
                    visualizations[r](ax[r], inputs, preds, all_preds)
            elif r in rowparams.keys():
                visualizations[r](ax[r], inputs, preds, **rowparams[r])
            else:
                visualizations[r](ax[r], inputs, preds)
            
            for c in range(cols):
                ax[r][c].set_xticks([], [])
                ax[r][c].set_yticks([], [])
            # use the following if you want to add a colorbar
            # if r == 0:
            #     fig.colorbar(ax[0][3], cax, orientation = 'vertical') #'ax0' tells it which plot to base the colors on

        if save_path is not None:
            fig.savefig(save_path)
        if show:
            plt.show()
        plt.close()
        return

    def vis_x_pred_y(self, ax, inputs, preds, vmin=0, vmax=1):
        """ Visualizes the moving image, the predicted fixed image, the ground truth fixed image and the flow field."""
        x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds

        ax[0].imshow(np.rot90(x[0,0]), "gray", vmin=vmin, vmax=vmax)
        ax[1].imshow(np.rot90(y_pred[0,0]), "gray", vmin=vmin, vmax=vmax)
        ax[2].imshow(np.rot90(y[0,0]), "gray", vmin=vmin, vmax=vmax)
        ax[3].imshow(np.rot90(flow_to_image(final_dfs[0]).permute(0,2,3,1)[0]))

        ax[0].set_xlabel("Input")
        ax[1].set_xlabel("Prediction")
        ax[2].set_xlabel("Target")
        ax[3].set_xlabel("DF")  
        ax[0].set_ylabel("input vs prediction")

        # colorbar
        cmap = get_cmap('hsv')
        cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax[3])
        cbar.set_ticks([0.18,0.51,0.7,1.0])
        cbar.set_ticklabels(["\u2190","\u2193","\u2192", "\u2191"])
        return
    
    def vis_segx_segpred_segy(self, ax, inputs, preds):
        """ Visualizes the moving segmentation, the predicted fixed segmentation, the ground truth fixed segmentation and the flow field."""
        x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds

        maxseg_x = seg_x[0].argmax(0)
        maxseg_y = seg_y[0].argmax(0)
        max_y_pred_seg = y_pred_seg[0].argmax(0)

        ax[0].imshow(np.rot90(maxseg_x))
        ax[1].imshow(np.rot90(max_y_pred_seg))
        ax[2].imshow(np.rot90(maxseg_y))
        ax[3].imshow(np.rot90(flow_to_image(df_pred).permute(0,2,3,1)[0]))

        ax[0].set_xlabel("Input")
        ax[1].set_xlabel("Prediction")
        ax[2].set_xlabel("Target")
        ax[3].set_xlabel("DF")
        ax[0].set_ylabel("segmentation input vs prediction")

        # colorbar
        cmap = get_cmap('hsv')
        cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax[3])
        cbar.set_ticks([0.18,0.51,0.7,1.0])
        cbar.set_ticklabels(["\u2190","\u2193","\u2192", "\u2191"])
        return
    
    def vis_pred_per_level(self, ax, inputs, preds, vmin=0, vmax=1):
        """ Visualizes the prediction of each output level of PULPo."""
        x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds
            
        for l in reversed(range(self.latent_levels)):
            ax[self.latent_levels-l-1].imshow(np.rot90(outputs[l][0,0]), "gray", vmin=vmin, vmax=vmax)
            ax[self.latent_levels-l-1].set_xlabel(f"Level {l}")
        ax[0].set_ylabel("Predictions per level")
        if self.latent_levels < 4:
            for l in range(self.latent_levels,4):
                ax[l].axis('off')
        return
    
    def vis_segpred_per_level(self, ax, inputs, preds):
        """ Visualizes the predicted segmentation of each output level of PULPo."""
        x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds
            
        for l in reversed(range(self.latent_levels)):
            ax[self.latent_levels-l-1].imshow(np.rot90(warped_seg[l][0].argmax(0)))
            ax[self.latent_levels-l-1].set_xlabel(f"Level {l}")
        ax[0].set_ylabel("Predicted segmentation per level")
        if self.latent_levels < 4:
            for l in range(self.latent_levels,4):
                ax[l].axis('off')
        return



    # plot the difference of the input and the prediction of each output level
    def vis_diff_input_pred(self, ax, inputs, preds, vmin=-1, vmax=1):
        """ Visualizes the difference of the input/moving image and the predicted fixed image of each output level."""
        x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds
            
        for l in reversed(range(self.latent_levels)):
            # use interpolate for x[0,0] to have the same size as outputs[l][0,0]
            ax[self.latent_levels-l-1].imshow(np.rot90(outputs[l][0,0] - torch.nn.functional.interpolate(x, size=outputs[l][0,0].shape, mode="bilinear", align_corners=False)[0,0]), "gray", vmin=vmin, vmax=vmax)
            ax[self.latent_levels-l-1].set_xlabel(f"Level {l}")
        ax[0].set_ylabel("Difference Input / Predictions per level")
        if self.latent_levels < 4:
            for l in range(self.latent_levels,4):
                ax[l].axis('off')
        return
    
    def vis_diff_target_pred(self, ax, inputs, preds, vmin=-1, vmax=1):
        """ Visualizes the difference of the target/fixed image and the predicted fixed image of each output level."""
        x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds
            
        for l in reversed(range(self.latent_levels)):
            # use interpolate for y[0,0] to have the same size as outputs[l][0,0]
            ax[self.latent_levels-l-1].imshow(np.rot90(outputs[l][0,0] - torch.nn.functional.interpolate(y, size=outputs[l][0,0].shape, mode="bilinear", align_corners=False)[0,0]), "gray", vmin=vmin, vmax=vmax)
            ax[self.latent_levels-l-1].set_xlabel(f"Level {l}")
        ax[0].set_ylabel("Difference Target / Predictions per level")
        if self.latent_levels < 4:
            for l in range(self.latent_levels,4):
                ax[l].axis('off')
        return

    def vis_jdet(self, ax, inputs, preds):
        """ Visualizes the jacobian determinant of the final deformation field for each level of PULPo."""
        x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds

        for l in reversed(range(self.latent_levels)):
            heatplot = sns.heatmap(np.rot90(jacobian_det(final_dfs[l])[0].detach()),
                                   ax=ax[self.latent_levels-l-1],
                                   cmap=sns.diverging_palette(10,250,sep=1,n=100),
                                   center=0., vmin=-2,vmax=4)
            # remove the legend for all but the last column
            if l != 0:
                heatplot.collections[0].colorbar.remove()
            ax[self.latent_levels-l-1].set_xlabel(f"Level {l}")
        
        # use to set colorbar for the last column
        # colorbar = plt.colorbar(heatplot.collections[0], ax=ax[4])
        ax[0].set_ylabel("heatmap of JDet std")
        if self.latent_levels < 4:
            for l in range(self.latent_levels,4):
                ax[l].axis('off')
        return
    
    def vis_final_df_per_level(self, ax, inputs, preds, flow=True, grid=True):
        """ Visualizes the final deformation field of each level of PULPo."""
        x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds

        for l in reversed(range(self.latent_levels)):
            if flow==True:
                ax[self.latent_levels-l-1].imshow(np.rot90(flow_to_image(final_dfs[l])[0].permute(1,2,0)))
            
            if grid == True:
                grid_i, grid_j = self.create_warped_grid(np.rot90(final_dfs[l],axes=(-2,-1)), self.grid_size)
                self.plot_grid(grid_i,grid_j, ax=ax[self.latent_levels-l-1], scatter=False, lines=True, straightlines=False, color="black", linewidth=0.5)
            ax[self.latent_levels-l-1].set_xlabel(f"Level {l}")
        ax[0].set_ylabel("Final DF per level.")
        if self.latent_levels < 4:
            for l in range(self.latent_levels,4):
                ax[l].axis('off')
        # use to also plot a colorbar
        # cmap = get_cmap('hsv')
        # cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax[3])
        # cbar.set_ticks([0.18,0.51,0.7,1.0])
        # cbar.set_ticklabels(["\u2190","\u2193","\u2192", "\u2191"])
        return

    def vis_combined_df_per_level(self, ax, inputs, preds, flow=True, grid=True):
        """ Visualizes the combined deformation field of each level of PULPo."""
        x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds

        for l in reversed(range(self.latent_levels)):
            if flow==True:
                ax[self.latent_levels-l-1].imshow(np.rot90(flow_to_image(combined_dfs[l])[0].permute(1,2,0)))
            
            if grid == True:
                grid_i, grid_j = self.create_warped_grid(np.rot90(combined_dfs[l],axes=(-2,-1)), self.grid_size)
                self.plot_grid(grid_i,grid_j, ax=ax[self.latent_levels-l-1], scatter=False, lines=True, straightlines=False, color="black", linewidth=0.5)
            ax[self.latent_levels-l-1].set_xlabel(f"Level {l}")
        ax[0].set_ylabel("Combined DF per level.")
        if self.latent_levels < 4:
            for l in range(self.latent_levels,4):
                ax[l].axis('off')

        # use to also plot a colorbar
        # cmap = get_cmap('hsv')
        # cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax[3])
        # cbar.set_ticks([0.18,0.51,0.7,1.0])
        # cbar.set_ticklabels(["\u2190","\u2193","\u2192", "\u2191"])
        return

    def vis_individual_df_per_level(self, ax, inputs, preds, flow=True, grid=True):
        """ Visualizes the individual deformation field of each level of PULPo."""
        x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds

        for l in reversed(range(self.latent_levels)):
            if flow==True:
                ax[self.latent_levels-l-1].imshow(np.rot90(flow_to_image(individual_dfs[l])[0].permute(1,2,0)))
            
            if grid == True:
                grid_i, grid_j = self.create_warped_grid(np.rot90(individual_dfs[l],axes=(-2,-1)), self.grid_size)
                self.plot_grid(grid_i,grid_j, ax=ax[self.latent_levels-l-1], scatter=False, lines=True, straightlines=False, color="black", linewidth=0.5)
            ax[self.latent_levels-l-1].set_xlabel(f"Level {l}")
        ax[0].set_ylabel("Individual DF per level.")
        if self.latent_levels < 4:
            for l in range(self.latent_levels,4):
                ax[l].axis('off')

        # use to also plot a colorbar
        # cmap = get_cmap('hsv')
        # cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax[3])
        # cbar.set_ticks([0.18,0.51,0.7,1.0])
        # cbar.set_ticklabels(["\u2190","\u2193","\u2192", "\u2191"])
        return
    
    def vis_output_var_per_level(self, ax, inputs, preds, all_preds):
        """ Visualizes the variance of the prediction of each output level of PULPo."""
        x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds
        output_std, individual_df_std, final_df_std, all_outputs, all_individual_dfs, all_combined_dfs, all_final_dfs, all_warped_seg = all_preds

        for l in reversed(range(self.latent_levels)):
            heatplot = sns.heatmap(np.rot90(output_std[l]**2), ax=ax[self.latent_levels-l-1])
            heatplot.collections[0].colorbar.remove()
            ax[self.latent_levels-l-1].set_xlabel(f"Level {l}")
        ax[0].set_ylabel("heatmap of prediction variance")
        if self.latent_levels < 4:
            for l in range(self.latent_levels,4):
                ax[l].axis('off')
        return
    
    def vis_individual_df_var_per_level(self, ax, inputs, preds, all_preds):
        """ Visualizes the variance of the individual deformation field of each level of PULPo."""
        x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds
        output_std, individual_df_std, final_df_std, all_outputs, all_individual_dfs, all_combined_dfs, all_final_dfs, all_warped_seg = all_preds

        for l in reversed(range(self.latent_levels)):
            # mask
            if l == 0:
                # reduces outputs[l]'s size by a factor of 2
                sth = torch.nn.functional.interpolate(outputs[l], size=(outputs[l].shape[-2]//2,outputs[l].shape[-1]//2), mode="bilinear", align_corners=False)
                individual_df_std[l][sth[0,0]==0] = 0
            else:
                individual_df_std[l][outputs[l][0,0]==0] = 0
            heatplot = sns.heatmap(np.rot90(individual_df_std[l]**2), ax=ax[self.latent_levels-l-1])
            heatplot.collections[0].colorbar.remove()
            ax[self.latent_levels-l-1].set_xlabel(f"Level {l}")
        ax[0].set_ylabel("heatmap of individual DF var")
        if self.latent_levels < 4:
            for l in range(self.latent_levels,4):
                ax[l].axis('off')
        return
    
    def vis_final_df_var_per_level(self, ax, inputs, preds, all_preds):
        """ Visualizes the variance of the final deformation field of each level of PULPo."""
        x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds
        output_std, individual_df_std, final_df_std, all_outputs, all_individual_dfs, all_combined_dfs, all_final_dfs, all_warped_seg = all_preds

        for l in reversed(range(self.latent_levels)):
            # mask
            final_df_std[l][outputs[l][0,0]==0] = 0
            heatplot = sns.heatmap(np.rot90(final_df_std[l]**2), ax=ax[self.latent_levels-l-1])
            heatplot.collections[0].colorbar.remove()
            ax[self.latent_levels-l-1].set_xlabel(f"Level {l}")
        ax[0].set_ylabel("heatmap of final DF var")
        if self.latent_levels < 4:
            for l in range(self.latent_levels,4):
                ax[l].axis('off')
        return
    
    def vis_sample_preds(self, ax, inputs, preds, all_preds, level=0, vmin=0, vmax=1):
        """ Visualizes sample predictions of PULPo."""
        x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds
        output_std, individual_df_std, final_df_std, all_outputs, all_individual_dfs, all_combined_dfs, all_final_dfs, all_warped_seg = all_preds

        for samp in range(all_outputs[level].shape[0] if all_outputs[level].shape[0] < self.latent_levels else self.latent_levels):
            ax[samp].imshow(np.rot90(all_outputs[level][samp,0]), "gray", vmin=0, vmax=1)
        ax[0].set_ylabel(f"Sample predictions on level {level}")
        return
    
    def vis_sample_segpreds(self, ax, inputs, preds, all_preds, level=0):
        """ Visualizes sample segmentations of PULPo."""
        x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds
        output_std, individual_df_std, final_df_std, all_outputs, all_individual_dfs, all_combined_dfs, all_final_dfs, all_warped_seg = all_preds

        for samp in range(all_warped_seg[level].shape[0] if all_warped_seg[level].shape[0] < self.latent_levels else self.latent_levels):
            ax[samp].imshow(np.rot90(all_warped_seg[level][samp].argmax(0)))
        ax[0].set_ylabel(f"Sample predicted segmentations on level {level}")
        return
    
    def vis_sample_dfs(self, ax, inputs, preds, all_preds, level=0, flow=True, grid=True):
        """ Visualizes sample deformation fields of PULPo."""
        x,y,seg_x,seg_y,lm_x,lm_y,mask_x,mask_y,loader = inputs
        y_pred, df_pred, y_pred_seg, outputs, individual_dfs, combined_dfs, final_dfs, warped_seg, prediction_name = preds
        output_std, individual_df_std, final_df_std, all_outputs, all_individual_dfs, all_combined_dfs, all_final_dfs, all_warped_seg = all_preds

        for samp in range(all_final_dfs[level].shape[0] if all_final_dfs[level].shape[0] < self.latent_levels else self.latent_levels):
            if grid == True:
                grid_i, grid_j = self.create_warped_grid(np.rot90(all_final_dfs[level][samp].unsqueeze(0),axes=(-2,-1)), self.grid_size)
                self.plot_grid(grid_i,grid_j, ax=ax[samp], scatter=True, lines=True, straightlines=False, color="black", linewidth=0.5)
            
        ax[0].set_ylabel(f"Sample predicted DFs on level {level}")

        # colorbar
        cmap = get_cmap('hsv')
        cbar = plt.colorbar(cm.ScalarMappable(cmap=cmap), ax=ax[3])
        cbar.set_ticks([0.18,0.51,0.7,1.0])
        cbar.set_ticklabels(["\u2190","\u2193","\u2192", "\u2191"])
        return


    ##################################################################################################################
    ########  big evaluation runs                    #################################################################
    ##################################################################################################################

    # Naive baseline model: affine registration
    def performance_affine(self, ndims, segs=False, lms=False, mask=False, output_dir="experiments/affine", artifact="", task="oasis"):
        with torch.no_grad():
            ndims = ndims
            self.load_data(task=task, segs=segs, lms=lms, mask=mask, ndims=ndims)
            
            os.makedirs(output_dir+"/"+task, exist_ok=True)

            #####################################################################################################
            ####### PERFORMANCE EVALUATION               ########################################################
            #####################################################################################################
            metric_names = ["RMSE"]
            metrics = [self.rmse]
            if segs:
                metric_names += ["Dice"]
                metrics.append(self.dsc)
            if lms:
                metric_names += ["LM_MAE"]
                metric_names += ["LM_Euclid"]
                metrics.append(self.lm_mae)
                metrics.append(self.lm_euclid)

            num_metrics = len(metric_names)
            
            all_metrics = np.zeros([num_metrics, self.num_datasets, self.num_inputs],dtype=float)

            for k, loader in enumerate(self.loaders): 
                print(f"Evaluating on {self.loader_names[k]}")
            
                # loop through all inputs
                for j, input in enumerate(loader):
                    if j % 50 == 0:
                        print(f"Input {j} of {len(loader)}")
                    x = input[0].to(self.device)
                    y = input[1].to(self.device)
                    seg_x = input[2].to(self.device)
                    seg_y = input[3].to(self.device)
                    lm_x = input[4].to(self.device)
                    lm_y = input[5].to(self.device)

                    if artifact != "":
                        x = self.artifact(x, method=artifact,x=(100,130), y=(100,130), z=(120,150))
                    
                    # calculate loss with all your metrics
                    losses = []
                    losses.append(self.rmse(x, y))
                    if loader.dataset.segs:
                        losses.append(self.dsc(seg_x, seg_y))
                    else:
                        if segs:
                            losses.append(0)
                    if loader.dataset.lms:
                        losses.append(self.lm_mae(lm_x, lm_y))
                        losses.append(self.lm_euclid(lm_x, lm_y))
                    else:
                        if lms:
                            losses.append(0)
                            losses.append(0)
                    for h,metric in enumerate(metrics):
                        all_metrics[h,k,j] = losses[h]


            # Not all datasets are of equal length, the mean has to ignore empty entries
            all_metrics[all_metrics == 0] = np.nan
            mean_metrics = np.nanmean(all_metrics, axis=-1)
            data = mean_metrics.T
            data = np.concatenate(data)[None,:]

            # create a pandas dataframe with hierarchical column names --> multiindex
            sets = np.repeat(self.loader_names,num_metrics)
            mets = np.tile(metric_names, self.num_datasets)
            mux = pd.MultiIndex.from_arrays((sets,mets))
            df = pd.DataFrame(data, columns=mux).round(6)
            # write dataframe to a latex table
            self.make_tables(df, output_dir=output_dir, name=task+"/loss_table_deterministic"+artifact)


    def performance_vxm(self, model_dir, name, artifact="", task="oasis", segs=False, lms=False):
        with torch.no_grad():
            model = self.load_vxm(model_dir,name)
            ndims = 3
            self.load_data(task=task, segs=segs, lms=lms, mask=False, ndims=ndims)

            os.makedirs("models/"+name+"/performance"+artifact, exist_ok=True)

            #####################################################################################################
            ####### PERFORMANCE EVALUATION               ########################################################
            #####################################################################################################
            num_metrics = 7
            metric_names = ["RMSE", "DSC", "JDetStd", "JDetMean", "JDetLeq0", "LM_MAE","TRE"]
            all_metrics = np.zeros([num_metrics, self.num_datasets, self.num_inputs],dtype=float) # [metrics, datasets, inputs]

            for k, loader in enumerate(self.loaders): 
                print(f"Evaluating on {self.loader_names[k]}")

                # if self.loader_names[k] != "test":
                #     all_metrics[:,k,:] = np.nan
                #     continue
            
                # loop through all inputs
                for j, input in enumerate(loader):
                    if j % 50 == 0:
                        print(f"Input {j} of {len(loader)}")
                    x = input[0].to(self.device)
                    y = input[1].to(self.device)
                    seg_x = input[2].to(self.device)
                    seg_y = input[3].to(self.device)
                    lm_x = input[4].to(self.device)
                    lm_y = input[5].to(self.device)

                    if artifact != "":
                        x = self.artifact(x, method=artifact,x=(100,130), y=(100,130), z=(120,150))
                    
                    y_pred, df_pred, _ = model(x, y, registration=True, deterministic=True)
                    # calculate loss with all your metrics
                    rmse = self.rmse(y_pred, y)
                    if loader.dataset.segs:
                        y_pred_seg = model.transformer(seg_x, df_pred)
                        dsc = self.dsc(y_pred_seg, seg_y)
                    else:
                        dsc = 0
                    jdet = self.jdet(df_pred)
                    jdetstd = jdet.std()
                    jdetmean = jdet.mean()
                    jdet_leq0_all = torch.where(jdet <= 0, 1, 0)
                    jdetleq0 = (torch.sum(jdet_leq0_all==1) / torch.sum(torch.ones_like(jdet_leq0_all)))*100
                    if loader.dataset.lms:
                        lm_pred = self.warp_landmarks(lm_x, df_pred)
                        lm_mae = self.lm_mae(lm_pred, lm_y)
                        tre = self.lm_euclid(lm_pred, lm_y)
                    else:
                        lm_mae = 0
                        tre = 0

                    all_metrics[0,k,j] = rmse
                    all_metrics[1,k,j] = dsc
                    all_metrics[2,k,j] = jdetstd
                    all_metrics[3,k,j] = jdetmean
                    all_metrics[4,k,j] = jdetleq0
                    all_metrics[5,k,j] = lm_mae
                    all_metrics[6,k,j] = tre
                    
            # Not all datasets are of equal length, the mean has to ignore empty entries
            all_metrics[all_metrics == 0] = np.nan
            mean_metrics = np.nanmean(all_metrics, axis=-1)
            data = mean_metrics.T
            data = np.concatenate(data)[None,:]

            # create a pandas dataframe with hierarchical column names --> multiindex
            sets = np.repeat(self.loader_names,num_metrics)
            mets = np.tile(metric_names, self.num_datasets)
            mux = pd.MultiIndex.from_arrays((sets,mets))
            df = pd.DataFrame(data, columns=mux).round(6)
            # write dataframe to a latex table
            self.make_tables(df, "models/"+name+"/performance", name="loss_table_deterministic"+artifact)

    def uncertainty_vxm(self, model_dir, name, num_samples, artifact="", task="oasis",lms=False):
        with torch.no_grad():
            model = self.load_vxm(model_dir,name)
            ndims = 3
            self.load_data(task=task, segs=False, lms=lms, mask=False, ndims=ndims)

            os.makedirs("models/"+name+"/uncertainty"+artifact, exist_ok=True)

            #####################################################################################################
            ####### UNCERTAINTY EVALUATION               ########################################################
            #####################################################################################################
            # 2 because we have 2 uncertainty metrics
            num_metrics = 4
            metric_names = ["Var", "NCC", "LM_VAR", "LM_NCC"]
            all_metrics = np.zeros([num_metrics, self.num_datasets, self.num_inputs],dtype=float) # [metrics, datasets, inputs]

            for k, loader in enumerate(self.loaders): 
                print(f"Evaluating on {self.loader_names[k]}")

                # if self.loader_names[k] != "test":
                #     all_metrics[:,k,:] = np.nan
                #     continue
            
                # loop through all inputs
                for j, input in enumerate(loader):
                    if j % 50 == 0:
                        print(f"Input {j} of {len(loader)}")
                    x = input[0].to(self.device)
                    y = input[1].to(self.device)
                    seg_x = input[2].to(self.device)
                    seg_y = input[3].to(self.device)
                    lm_x = input[4].to(self.device)
                    lm_y = input[5].to(self.device)

                    if artifact != "":
                        x = self.artifact(x, method=artifact,x=(100,130), y=(100,130), z=(120,150))

                    preds, all_preds = self.predict_vxm(x, y, num_samples=num_samples, deterministic=False)
                    
                    moved_std = all_preds[0]
                    all_moved = all_preds[3]

                    # calculate loss with all your metrics
                    mse = np.array(torch.mean((all_moved - y)**2, axis=0).squeeze(0).to("cpu"))
                    var = (moved_std**2)
                    var = np.array(var.to("cpu"))
                    ncc = self.ncc(var, mse)
                    var = var.mean()
                    
                    # calculate lm metrics
                    if loader.dataset.lms:
                        df_pred = preds[1]
                        lm_hat = self.warp_landmarks(lm_x, df_pred)
                        all_df_pred = all_preds[6]
                        warped_lms = self.warp_landmarks(lm_x, all_df_pred)
                        lm_var = self.lms_var(warped_lms).mean()
                        lm_ncc = self.lms_corr(lm_hat, warped_lms, lm_y)
                    else:
                        lm_var = 0
                        lm_ncc = 0

                    all_metrics[0,k,j] = var
                    all_metrics[1,k,j] = ncc
                    all_metrics[2,k,j] = lm_var
                    all_metrics[3,k,j] = lm_ncc
                    
            # Not all datasets are of equal length, the mean has to ignore empty entries
            all_metrics[all_metrics == 0] = np.nan
            mean_metrics = np.nanmean(all_metrics, axis=-1)
            data = mean_metrics.T
            data = np.concatenate(data)[None,:]

            # create a pandas dataframe with hierarchical column names --> multiindex
            sets = np.repeat(self.loader_names,num_metrics)
            mets = np.tile(metric_names, self.num_datasets)
            mux = pd.MultiIndex.from_arrays((sets,mets))
            df = pd.DataFrame(data, columns=mux)
            print(f"Uncertainty evaluation on {name}", df)
            # write dataframe to a latex table
            self.make_tables(df, "models/"+name+"/uncertainty", name="loss_table"+artifact)

    def performance(self, model_dir, git_hash, version, segs, lms, mask, task="oasis", artifact=""):
        with torch.no_grad():
            model = self.load_model(model_dir=model_dir, git_hash=git_hash, version=version)
            ndims = model.ndims
            self.load_data(task=task, segs=segs, lms=lms, mask=mask, ndims=ndims)

            os.makedirs(self.output_dir+"/loss", exist_ok=True)
            
            all_metrics = np.zeros([self.num_metrics, self.latent_levels, self.num_datasets, self.num_inputs],dtype=float) # [metrics, datasets, inputs]

            # initiate the leveled losses with neutral weight dictionaries
            hierarchical_metrics = []
            if "RMSE" in self.metric_names:
                hierarchical_mse = HierarchicalReconstructionLoss(["mse"],{l:1.0 for l in range(self.latent_levels)}, similarity_pyramid=False, ndims=ndims, window_size={l:1.0 for l in range(self.latent_levels)})
                hierarchical_metrics.append(hierarchical_mse)
            if "JDetStd" in self.metric_names:
                hierarchical_jdet_std = HierarchicalRegularization(JDetStd,{l:1.0 for l in range(self.latent_levels)}, similarity_pyramid=False)
                hierarchical_metrics.append(hierarchical_jdet_std)
            if "Dice" in self.metric_names:
                hierarchical_dice = HierarchicalReconstructionLoss(["dice"],{l:1.0 for l in range(self.latent_levels)}, similarity_pyramid=False, ndims=ndims, window_size={l:1.0 for l in range(self.latent_levels)})
                hierarchical_metrics.append(hierarchical_dice)

        
            for k, loader in enumerate(self.loaders): 
                print(f"Evaluating on {self.loader_names[k]}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
                # loop through all inputs
                for j, input in enumerate(loader):
                    # HACK: only evaluate on 1 input, remove later
                    # if j == 1:
                    #     break
                    if j % 50 == 0:
                        print(f"Input {j} of {len(loader)}")
                    x = input[0].to(self.device)
                    y = input[1].to(self.device)
                    seg_x = input[2].to(self.device)
                    seg_y = input[3].to(self.device)
                    lm_x = input[4].to(self.device)
                    lm_y = input[5].to(self.device)


                    if artifact != "":
                        x = self.artifact(x, method=artifact,x=(100,130), y=(100,130), z=(120,150))
                    

                    # outputs, individual_dfs = model.predict(x,y,N)
                    outputs, individual_dfs = model.predict_deterministic(x,y)
                    _, final_dfs = model.combine_dfs(individual_dfs)

                    if loader.dataset.segs:
                        pred_segs = {key: model.autoencoder.decoders[key].spatial_transform(final_dfs[key], seg_x) for key in range(self.latent_levels)}
                    else:
                        pred_segs = None
                        
                    
                    # we want to get pixelwise losses, so we will share some losses by the number of pixels
                    num_pixels = {l: torch.prod(torch.tensor(outputs[l].size()[2:])) for l in range(self.latent_levels)}
                    # calculate loss with all your metrics
                    level_losses = []
                    if "RMSE" in self.metric_names:
                        mse, level_mse = hierarchical_mse(y_hat = outputs,y = y, y_hat_seg=pred_segs, seg_y=seg_y, ncc_factor = 1, dice_factor = 1)
                        level_rmse = {key: torch.sqrt(level_mse[key] / num_pixels[key]) for key in level_mse.keys()}
                        level_losses.append(level_rmse)
                    if "JDetStd" in self.metric_names:
                        jdet, level_jdet = hierarchical_jdet_std(final_dfs, lamb=1)
                        level_losses.append(level_jdet)
                    if "JDetLeq0" in self.metric_names:
                        level_jdetleq0 = {key: 0 for key in range(self.latent_levels)}
                        for key in range(self.latent_levels):
                            jdet = jacobian_det(final_dfs[key])
                            # take product of all dimensions
                            level_jdetleq0[key] = (torch.sum(jdet <= 0) / torch.prod(torch.tensor(jdet.squeeze().size())))*100
                        level_losses.append(level_jdetleq0)
                    if "Dice" in self.metric_names:
                        if not loader.dataset.segs:
                            level_losses.append({key: 0 for key in range(self.latent_levels)})
                        else:
                            dice, level_dice = hierarchical_dice(y_hat = outputs,y = y, y_hat_seg=pred_segs, seg_y=seg_y, ncc_factor = 1, dice_factor = 1)
                            level_dice = {key: 1 - (level_dice[key] / num_pixels[key]) for key in level_dice.keys()}
                            level_losses.append(level_dice)
                    if "LM_MAE" in self.metric_names:
                        if not loader.dataset.lms:
                            level_losses.append({key: 0 for key in range(self.latent_levels)})
                        else:
                            # HACK: we don't have any levels yet
                            level_lm_mae = {key: 0 for key in range(self.latent_levels)}
                            # Make sure that the landmarks are not empty
                            if lm_x.numel() and lm_y.numel():
                                level_lm_mae[0] = self.lm_mae(model.warp_landmarks(lm_x, final_dfs[0]).detach(), lm_y)
                            level_losses.append(level_lm_mae)
                    if "LM_Euclid" in self.metric_names:
                        if not loader.dataset.lms:
                            level_losses.append({key: 0 for key in range(self.latent_levels)})
                        else:
                            level_lm_euclid = {key: 0 for key in range(self.latent_levels)}
                            # Make sure that the landmarks are not empty
                            if lm_x.numel() and lm_y.numel():
                                level_lm_euclid[0] = self.lm_euclid(model.warp_landmarks(lm_x, final_dfs[0]).detach(), lm_y)
                            level_losses.append(level_lm_euclid)
                    
                    for h, level_loss in enumerate(level_losses):
                        for l in range(self.latent_levels):
                            all_metrics[h,l,k,j] = level_loss[l]

            # Not all datasets are of equal length, the mean has to ignore empty entries
            all_metrics[all_metrics == 0] = np.nan
            mean_metrics = np.nanmean(all_metrics, axis=-1)
            data = mean_metrics.T
            data = np.concatenate(data, axis=1)

            # create a pandas dataframe with hierarchical column names --> multiindex
            sets = np.repeat(self.loader_names,self.num_metrics)
            mets = np.tile(self.metric_names, self.num_datasets)
            mux = pd.MultiIndex.from_arrays((sets,mets))
            df = pd.DataFrame(data, columns=mux,index=range(self.latent_levels)).round(3)
            # write dataframe to a latex table
            self.make_tables(df, self.output_dir, name="loss/loss_table_deterministic")

            # explicit garbage collection
            x=y=seg_x=seg_y=None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def uncertainty(self, model_dir, git_hash, version, segs, lms, mask, num_samples, artifact="", task="oasis"):
        if num_samples < 2:
            raise ValueError("N has to be at least 2")
        with torch.no_grad():
            model = self.load_model(model_dir=model_dir, git_hash=git_hash, version=version)
            ndims = model.ndims
            self.load_data(task=task, segs=segs, lms=lms, mask=mask, ndims=ndims)

            os.makedirs(self.output_dir+"/uncertainty", exist_ok=True)

            metric_names = ["Var", "NCC"]
            if lms:
                metric_names += ["LM_VAR", "LM_NCC"]
            num_metrics = len(metric_names)
            all_metrics = np.zeros([num_metrics, self.num_datasets, self.num_inputs],dtype=float) # [metrics, datasets, inputs]

            for k, loader in enumerate(self.loaders): 
                print(f"Evaluating on {self.loader_names[k]}")

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                # loop through all inputs
                for j, input in enumerate(loader):
                    if j % 50 == 0:
                        print(f"Input {j} of {len(loader)}")
                    
                    x,y,seg1,seg2,lm1,lm2,mask_x,mask_y = input
                    input = [x.to(self.device), y.to(self.device), seg1.to(self.device), seg2.to(self.device), lm1.to(self.device), lm2.to(self.device), mask_x.to(self.device), mask_y.to(self.device), self.loader_names[k]]
                    x,y,seg1,seg2,lm1,lm2,mask_x,mask_y,_ = input
                    preds, all_preds = self.predict(input, num_samples=num_samples, deterministic=False)

                    moved_std = all_preds[0][0]
                    all_moved = all_preds[3][0]

                    # calculate loss with all your metrics
                    mse = np.array(torch.mean((all_moved - y)**2, axis=0).squeeze(0).to("cpu"))
                    var = (moved_std**2)
                    var = np.array(var.to("cpu"))
                    ncc = self.ncc(var, mse)
                    var = var.mean()

                    all_metrics[0,k,j] = var
                    all_metrics[1,k,j] = ncc

                    if loader.dataset.lms:
                        # warp the landmarks
                        df_pred = preds[1]
                        lm_hat = self.warp_landmarks(lm1, df_pred)
                        all_df_pred = all_preds[6][0]
                        warped_lms = model.warp_landmarks(lm1, all_df_pred)
                        lm_var = self.lms_var(warped_lms).mean()
                        lm_ncc = self.lms_corr(lm_hat, warped_lms, lm2)
                        all_metrics[2,k,j] = lm_var.mean()
                        all_metrics[3,k,j] = lm_ncc
                    
            # Not all datasets are of equal length, the mean has to ignore empty entries
            all_metrics[all_metrics == 0] = np.nan
            mean_metrics = np.nanmean(all_metrics, axis=-1)
            data = mean_metrics.T
            data = np.concatenate(data)[None,:]

            # create a pandas dataframe with hierarchical column names --> multiindex
            sets = np.repeat(self.loader_names,num_metrics)
            mets = np.tile(metric_names, self.num_datasets)
            mux = pd.MultiIndex.from_arrays((sets,mets))
            df = pd.DataFrame(data, columns=mux)
            # write dataframe to a latex table
            self.make_tables(df, self.output_dir, name="uncertainty/loss_table")

            # explicit garbage collection
            x=y=None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


    def run_one_model(self, model_dir, git_hash, version, segs, lms, mask, N=10, task="oasis"):
        with torch.no_grad():
            model = self.load_model(model_dir=model_dir, git_hash=git_hash, version=version)
            ndims = model.ndims
            self.load_data(task=task, segs=segs, lms=lms, mask=mask, ndims=ndims)


            os.makedirs(self.output_dir+"/jdet", exist_ok=True)
            os.makedirs(self.output_dir+"/loss", exist_ok=True)
            os.makedirs(self.output_dir+"/uncertainty", exist_ok=True)
            os.makedirs(self.output_dir+"/vis", exist_ok=True)

            #####################################################################################################
            ####### FIRST ALL VISUALIZATIONS               ######################################################
            #####################################################################################################
            print("RUNNING VISUALIZATIONS")
            for k,l in enumerate(self.loader_names):
                self.segs = self.loaders[k].dataset.segs
                inputs = self.sample_data(loader_name=l)

                predict_methods = ["deterministic", "sample", f"3Davg_{N}"] if ndims == 3 else ["deterministic", "sample", f"avg_{N}"]
                for p in predict_methods:
                    if p == "deterministic":
                        preds, all_preds = self.predict(inputs, num_samples=1, deterministic=True)
                    elif p == "sample":
                        preds, all_preds = self.predict(inputs, num_samples=1, deterministic=False)
                    elif p == f"avg_{N}":
                        preds, all_preds = self.predict(inputs, num_samples=N, deterministic=False)
                    elif p == f"3Davg_{N}":
                        preds, all_preds = self.predict(inputs, num_samples=N, deterministic=False)

                    self.grid_size = 20
                    if all_preds == []:
                        if self.loaders[k].dataset.segs:
                            self.visualize(inputs, preds,[self.vis_x_pred_y,
                                                self.vis_segx_segpred_segy,
                                                self.vis_pred_per_level,
                                                self.vis_segpred_per_level,
                                                self.vis_diff_input_pred,
                                                self.vis_diff_target_pred,
                                                self.vis_final_df_per_level,
                                                self.vis_combined_df_per_level,
                                                self.vis_individual_df_per_level,
                                                self.vis_jdet,
                                                    ],
                                                all_preds=all_preds,
                                                    title=f"All visualizations on {l} set with {p} prediction", show=False, save_path=self.output_dir + f"/vis/allvis{l}_{p}.png")
                        else:
                            self.visualize(inputs, preds,[self.vis_x_pred_y,
                                                self.vis_pred_per_level,
                                                self.vis_diff_input_pred,
                                                self.vis_diff_target_pred,
                                                self.vis_final_df_per_level,
                                                self.vis_combined_df_per_level,
                                                self.vis_individual_df_per_level,
                                                self.vis_jdet,
                                                    ],
                                                all_preds=all_preds,
                                                    title=f"All visualizations on {l} set with {p} prediction", show=False, save_path=self.output_dir + f"/vis/allvis{l}_{p}.png")
                    elif p == f"3Davg_{N}":
                        self.visualize(inputs, preds,[self.vis_x_pred_y,
                                            self.vis_pred_per_level,
                                            self.vis_diff_input_pred,
                                            self.vis_diff_target_pred,
                                            self.vis_final_df_per_level,
                                            self.vis_combined_df_per_level,
                                            self.vis_individual_df_per_level,
                                            self.vis_jdet,
                                            self.vis_output_var_per_level,
                                            self.vis_individual_df_var_per_level,
                                            self.vis_final_df_var_per_level,
                                            self.vis_sample_preds,
                                            self.vis_sample_dfs,
                                            ],
                                            all_preds=all_preds,
                                                title=f"All visualizations on {l} set with {p} prediction", show=False, save_path=self.output_dir + f"/vis/allvis{l}_{p}.png")
                                                    
                    else:
                        if self.loaders[k].dataset.segs:
                            self.visualize(inputs, preds,[self.vis_x_pred_y,
                                                self.vis_segx_segpred_segy,
                                                self.vis_pred_per_level,
                                                self.vis_segpred_per_level,
                                                self.vis_diff_input_pred,
                                                self.vis_diff_target_pred,
                                                self.vis_final_df_per_level,
                                                self.vis_combined_df_per_level,
                                                self.vis_individual_df_per_level,
                                                self.vis_jdet,
                                                self.vis_output_var_per_level,
                                                self.vis_individual_df_var_per_level,
                                                self.vis_final_df_var_per_level,
                                                self.vis_sample_preds,
                                                self.vis_sample_segpreds,
                                                self.vis_sample_dfs,
                                                    ],
                                                all_preds=all_preds,
                                                rowparams={6:{"flow":True, "grid":False},
                                                            7:{"flow":True, "grid":False}},
                                                    title=f"All visualizations on {l} set with {p} prediction", show=False, save_path=self.output_dir + f"/vis/allvis{l}_{p}.png")
                        else:
                            self.visualize(inputs, preds,[self.vis_x_pred_y,
                                                self.vis_pred_per_level,
                                                self.vis_diff_input_pred,
                                                self.vis_diff_target_pred,
                                                self.vis_final_df_per_level,
                                                self.vis_combined_df_per_level,
                                                self.vis_individual_df_per_level,
                                                self.vis_jdet,
                                                self.vis_output_var_per_level,
                                                self.vis_individual_df_var_per_level,
                                                self.vis_final_df_var_per_level,
                                                self.vis_sample_preds,
                                                self.vis_sample_dfs,
                                                    ],
                                                all_preds=all_preds,
                                                    title=f"All visualizations on {l} set with {p} prediction", show=False, save_path=self.output_dir + f"/vis/allvis{l}_{p}.png")
                    plt.close()
                    self.table_jdet(inputs, preds, output_dir=self.output_dir+"/jdet", name=f"{l}_{p}", show=False, save=True, fontsize=10)

            # explicit garbace collection
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            inputs =preds=all_preds=None
            gc.collect()

            #####################################################################################################
            ####### PERFORMANCE EVALUATION               ########################################################
            #####################################################################################################
            self.performance(model_dir, git_hash, version, segs, lms, mask, task=task)

            #####################################################################################################
            ####### UNCERTAINTY EVALUATION               ########################################################
            #####################################################################################################
            if N > 1:
                self.uncertainty(model_dir, git_hash, version, segs, lms, mask, num_samples=N, task=task)
            else:
                print("N<=1, so no uncertainty evaluation.")
            return
    
    def compare_models(self, model_dir, exp_name, models, ndims, task="oasis", segs=True, model_names=None, N=10):
        #####################################################################################################
        ####### NECESSARY STUFF           ###################################################################
        #####################################################################################################
        
        output_dir = "experiments/"+exp_name
        # create the path if it doesn't exists yet
        os.makedirs(output_dir, exist_ok=True)
        
        self.models = models
        self.num_models = len(models)
        if model_names is not None:
            if self.num_models == len(model_names):
                self.model_names = model_names
            else:
                raise ValueError("model and model_names do not have same length!")
        else:
            self.model_names = self.models

        self.load_data(task=task, segs=self.segs, lms=self.lms, mask=self.mask, ndims=ndims)



        #####################################################################################################
        ####### RUN MODELS              #####################################################################
        #####################################################################################################

        all_metrics = np.zeros([self.num_metrics, self.num_models, self.num_datasets, self.num_inputs],dtype=float)
        # loop through all models"oasis"
        for i,m in enumerate(models):
            print(f"running model {i}")
            checkpoint = self.build_path(model_dir,m)
            self.load_model(model_dir=model_dir, git_hash=m.split("/")[0], version=m.split("/")[1])

            model = PULPo.load_from_checkpoint(checkpoint)
            model.eval()
            self.model = model
            self.latent_levels = model.latent_levels
            for k, loader in enumerate(self.loaders):
                # loop through all inputs
                for j, input in enumerate(loader):
                    x = input[0].to(self.device)
                    y = input[1].to(self.device)
                    seg_x = input[2].to(self.device)
                    seg_y = input[3].to(self.device)

                    outputs, dfs = model.predict(x,y,N)

                    _, final_dfs = model.combine_dfs(dfs)

                    y_pred = outputs[0]
                    y_pred_seg = model.autoencoder.decoders[0].spatial_transform(final_dfs[0], seg_x)
                    df_pred = final_dfs[0]

                    num_pixels = torch.prod(torch.tensor(outputs[0].size()[2:]))
                    # calculate loss with all your metrics
                    for h, metric in enumerate(self.metrics):
                        if self.metric_names[h] == "Dice":
                            all_metrics[h,i,k,j] = 1 - metric(y_pred_seg,seg_y, dice_factor=1) / num_pixels
                        elif self.metric_names[h] == "JDetStd":
                            all_metrics[h,i,k,j] = metric(df_pred,lamb=1)
                        else:
                            all_metrics[h,i,k,j] = torch.sqrt(metric(y_pred,y) / num_pixels)


        # Not all datasets are of equal length, the mean has to ignore empty entries
        all_metrics[all_metrics == 0] = np.nan
        mean_metrics = np.nanmean(all_metrics, axis=-1)
        data = mean_metrics.T
        data = np.concatenate(data, axis=1)

        # create a pandas dataframe with hierarchical column names --> multiindex
        sets = np.repeat(self.loader_names,self.num_metrics)
        mets = np.tile(self.metric_names, self.num_datasets)
        mux = pd.MultiIndex.from_arrays((sets,mets))
        df = pd.DataFrame(data, columns=mux, index=self.model_names).round(3)
        # write dataframe to a latex table
        self.make_tables(df, output_dir, name="loss_table")

        return




if __name__ == '__main__':

    #####################################################################################################
    ####### PARSE ARGUMENTS           ###################################################################
    #####################################################################################################

    parser = argparse.ArgumentParser(description="Main trainer file for all models.")
    parser.add_argument("--model_dir", type=str,
                            help="Provide relative super-directory of the models.",
                            required=True)
    parser.add_argument("--git_hash", type=str,
                            help="The git-hash used to run the model.",
                            required=True)
    parser.add_argument("--version", type=str,
                            help="The version of the model under the same git-hash.",
                            required=True)
    parser.add_argument('--segs', dest='segs', action='store_true',default=False)
    parser.add_argument('--lms', dest='lms', action='store_true',default=False)
    parser.add_argument('--mask', dest='mask', action='store_true',default=False)
    parser.add_argument('--task', default="oasis", type=str, required=False)
    parser.add_argument('--N', default=1, type=int, required=False)
    args = parser.parse_args()
    # loop through args and print them
    for arg in vars(args):
        print(arg, getattr(args, arg))

    eval = Evaluate()
    eval.run_one_model(model_dir=args.model_dir,
                       git_hash=args.git_hash,
                       version=args.version,
                       segs=args.segs,
                       lms=args.lms,
                       mask=args.mask,
                       N=args.N,
                       task=args.task)

    # # for model comparison:
    # parser = argparse.ArgumentParser(description="Main trainer file for all models.")
    # parser.add_argument("--model_dir", type=str,
    #                         help="Provide relative super-directory of the models.",
    #                         required=True)
    # parser.add_argument("--exp_name", type=str,
    #                         help="Provide one-word description of experiment.",
    #                         required=True)
    # parser.add_argument("--models", nargs="+", type=str,
    #                         help="Provide all models you want to compare. Expected format: git_hash/version. In your case anything between the model directory and the checkpoint folder.",
    #                         required=True)
    # parser.add_argument("--ndims", type=int,required=True)
    # parser.add_argument("--model_names", nargs="+", type=str,
    #                         help="Name your models. Optional.",
    #                         required=False)
    # args = parser.parse_args()

    # eval = Evaluate()
    # eval.compare_models(model_dir= args.model_dir, exp_name= args.exp_name, models= args.models, ndims=args.ndims, task="oasis", segs=True, model_names=args.model_names)