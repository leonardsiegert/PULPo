import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import subprocess
import argparse

from icecream import install
install()
#ic.disable()

from src.data import mini_reg
from src.data.OASIS import oasis
from src.data.BraTS import brats
from src.data.LM_OASIS import lm_oasis
from src.models import PHIReg
from evaluate import Evaluate

#################################################################################################################################################################################
##########  HYPERPARAMETERS                   ###################################################################################################################################
#################################################################################################################################################################################
accelerator = "cpu"
dataset = "oasis"
segs = False
lms = False
mask = False
total_levels = 6
latent_levels = 4
beta = 1
batch_size = 12
learning_rate = 1e-4
recon_loss = ["mse"]
dice_factor=50
ncc_factor=100
similarity_pyramid = False
lamb = 0
regularizer = "jdet"
image_logging_frequency = 1000
decoder = "SVF" # "BSpline"
feedback = ["samples"]
df_resolution = "full_res" # "full_res" or "level_res"
df_combination = "add" # "add" currently only option
ndims = 3



def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def main(hparams):
    print(hparams)
    print()

    git_hash = get_git_revision_short_hash()
    human_readable_extra = ""
    experiment_name = "-".join(
        [git_hash, f"seed={hparams.random_seed}", human_readable_extra]
    )

    pl.seed_everything(seed=hparams.random_seed)

    if hparams.segs and "dice" not in hparams.recon_loss:
        raise ValueError("You are trying to load segmentations but dice is not in the recon_loss.")
    
    if hparams.dataset == "brats":
        (
            train_loader,
            validation_loader,
            test_loader,
        ) = brats.create_data_loaders(batch_size=hparams.batch_size, 
                                        segs=hparams.segs, 
                                        lms=hparams.lms, 
                                        mask=hparams.mask,
                                        ndims=hparams.ndims,
                                        interpatient=hparams.interpatient)
    elif hparams.dataset == "oasis":
        (
                train_loader,
                test_loader,
                validation_loader,
            ) = oasis.create_train_test_val_loaders(batch_size=hparams.batch_size, 
                                                    segs=hparams.segs, 
                                                    lms=hparams.lms, 
                                                    mask=hparams.mask, 
                                                    ndims=hparams.ndims)
    elif hparams.dataset == "lm_oasis":
        (
            train_loader,
            validation_loader, 
            test_loader_seg, 
            test_loader_lm
             ) = lm_oasis.create_data_loaders(batch_size=hparams.batch_size, 
                                                segs=hparams.segs, 
                                                lms=False, 
                                                mask=False, 
                                                ndims=hparams.ndims)
    else:
        raise ValueError("Dataset not recognized.")

    input_size = next(iter(train_loader))[0].shape[2:]

    model = PHIReg(segs=hparams.segs, lms=hparams.lms, mask=hparams.mask, nondiagonal=hparams.nondiagonal, cp_depth=hparams.cp_depth,
                   total_levels=hparams.total_levels, latent_levels=hparams.latent_levels, zdim=2, input_size=input_size,
                   beta=hparams.beta, lr=hparams.learning_rate, recon_loss=hparams.recon_loss, dice_factor=hparams.dice_factor,
                   ncc_factor=hparams.ncc_factor, similarity_pyramid= hparams.similarity_pyramid, lamb=hparams.lamb, regularizer=hparams.regularizer,
                   image_logging_frequency=hparams.image_logging_frequency, decoder=hparams.decoder, feedback=hparams.feedback,
                   df_resolution=hparams.df_resolution, df_combination=hparams.df_combination, n0=hparams.n0)


    logger = TensorBoardLogger(
        save_dir="./runs", name=experiment_name, default_hp_metric=False
    )
    checkpoint_callbacks = [
        ModelCheckpoint(
            monitor="val/total_loss",
            filename="best-total-loss-{epoch}-{step}",
        ),
        ModelCheckpoint(
            monitor="val/reconstruction_loss",
            filename="best-reconstruction-loss-{epoch}-{step}",
        )
    ]

    print(f"RUNNING FOR {hparams.max_epochs} EPOCHS.")

    trainer = pl.Trainer(
        logger=logger,
        val_check_interval=0.1,
        log_every_n_steps=5,
        accelerator=hparams.accelerator,
        devices=hparams.devices,
        callbacks=checkpoint_callbacks,
        max_epochs= hparams.max_epochs,
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader
    )

    print("STARTING EVALUATION.")
    eval = Evaluate()
    eval.run_one_model(model_dir="runs", 
                        git_hash=experiment_name, 
                        version="version_"+str(logger.version),
                        segs=hparams.segs,
                        lms=hparams.lms,
                        mask=hparams.mask,
                        N=10,
                        task=hparams.dataset)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Main trainer file for all models.")
    parser.add_argument(
        "--random_seed",
        dest="random_seed",
        action="store",
        default=0,
        type=int,
        help="Random seed for pl.seed_everything function.",
    )
    parser.add_argument("--devices", type=int, default=None)
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--accelerator", type=str, default=accelerator)
    parser.add_argument("--dataset", type=str, default=dataset, help="Dataset to use. Default is oasis. Alternatives: brats, lm_oasis.")
    parser.add_argument("--segs", action='store_true', default=segs, help="Do we load segmentations from the dataset.")
    parser.add_argument("--lms", action='store_true', default=lms, help="Do we load landmarks from the dataset.")
    parser.add_argument("--mask", action='store_true', default=mask, help="Do we load masks from the dataset.")
    parser.add_argument("--total_levels", type=int, default=total_levels)
    parser.add_argument("--latent_levels", type=int, default=latent_levels)
    parser.add_argument("--beta", type=float, default=beta)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--learning_rate", type=float, default=learning_rate)
    parser.add_argument("--recon_loss", nargs='+', default=recon_loss, help="Losses used in training. Default is mse. Options: mse, ncc, dice.")
    parser.add_argument("--dice_factor", type=int, default=dice_factor, help="Factor to scale dice up to MSE magnitude.")
    parser.add_argument("--ncc_factor", type=int, default=ncc_factor, help="Factor to scale ncc up to MSE magnitude.")
    parser.add_argument("--similarity_pyramid", action='store_true', default=similarity_pyramid, help="Whether to use a similarity pyramid or not.")
    parser.add_argument("--lambda", type=float, default=lamb, dest="lamb", help="Lambda of regularization. Setting to 0 equals no regularization.")
    parser.add_argument("--regularizer", type=str, default=regularizer, help="Regularizer to use. Default is jdet. Alternatives: L2.")
    parser.add_argument("--image_logging_frequency", type=int, default=image_logging_frequency)
    parser.add_argument("--decoder", type=str, default=decoder, help="Decoder to use. Default is BSpline. Alternative is SVF.")
    parser.add_argument("--feedback", nargs='+', default=feedback, help="Feedback connection between sampling layers. Default is combined_df. Options: samples, control_points, individual_dfs, combined_dfs, final_dfs, transformed.")
    parser.add_argument("--df_resolution", type=str, default=df_resolution, help="Whether the dfs and thus transformed images are created at the resolution of 2x the sampling or at full resolution. Options: full_res, level_res.")
    parser.add_argument("--df_combination", type=str, default=df_combination, help="Method used to combine dfs. Options: add.")
    parser.add_argument("--n0", type=int, default=batch_size)
    parser.add_argument("--ndims", type=int, default=ndims, help="Choose here if you want to work with volumes (3) or slices (2). Default is 2.")
    parser.add_argument("--interpatient", action='store_true', default=False, help="Whether to use the interpatient dataset or not. Only relevant for the BraTS dataset.")
    parser.add_argument("--nondiagonal", action='store_true', default=False, help="Whether to use the nondiagonal prior and KL or not.")
    parser.add_argument("--cp_depth", type=int, default=3, help="Depth of the control point layer. Default is 3.")
    
    args = parser.parse_args()

    main(args)