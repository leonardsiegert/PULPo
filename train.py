import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import subprocess
import argparse

from src.data.BraTS import brats
from src.data.OASIS import oasis
from src.models import PULPo
from evaluate import Evaluate

#################################################################################################################################################################################
##########  HYPERPARAMETERS                   ###################################################################################################################################
#################################################################################################################################################################################
# These are the hyperparameters used in the training of the model used for the paper.
accelerator = "gpu"
dataset = "brats" # "brats" or "oasis"
segs = False
lms = False
mask = False
feedback = ["samples", "velocity_field", "individual_dfs", "combined_dfs", "final_dfs", "transformed"]
df_resolution = "level_res"
ndims = 3
batch_size = 1
total_levels = 5
latent_levels = 4
beta = 0.1
learning_rate = 1e-4
recon_loss = ["ncc"]
gamma=0.05
lamb = 0.025
regularizer = "L2"
similarity_pyramid = False
image_logging_frequency = 5000


# To save the checkpoint with current git hash
def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )

def main(hparams):
    pl.seed_everything(seed=hparams.random_seed)

    # Create an experiment name based on the git hash and hyperparameters
    git_hash = get_git_revision_short_hash()
    human_readable_extra = ""
    experiment_name = "-".join(
        [git_hash, f"seed={hparams.random_seed}", human_readable_extra]
    )

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
            validation_loader, 
            test_loader_seg, 
            test_loader_lm
             ) = oasis.create_data_loaders(batch_size=hparams.batch_size, 
                                                segs=hparams.segs, 
                                                lms=False, 
                                                mask=False, 
                                                ndims=hparams.ndims)
    else:
        raise ValueError("Dataset not recognized.")

    input_size = next(iter(train_loader))[0].shape[2:]

    model = PULPo(segs=hparams.segs, lms=hparams.lms, mask=hparams.mask, nondiagonal=hparams.nondiagonal, cp_depth=hparams.cp_depth,
                   total_levels=hparams.total_levels, latent_levels=hparams.latent_levels, input_size=input_size,
                   beta=hparams.beta, lr=hparams.learning_rate, recon_loss=hparams.recon_loss, dice_factor=hparams.dice_factor,
                   gamma=hparams.gamma, similarity_pyramid= hparams.similarity_pyramid, lamb=hparams.lamb, regularizer=hparams.regularizer,
                   image_logging_frequency=hparams.image_logging_frequency, feedback=hparams.feedback,
                   df_resolution=hparams.df_resolution, n0=hparams.n0)


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
        callbacks=checkpoint_callbacks,
        max_epochs= hparams.max_epochs,
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=validation_loader
    )

    print("TRAINING FINISHED, STARTING EVALUATION.")
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
    parser.add_argument("--max_epochs", type=int, default=1000)
    parser.add_argument("--accelerator", type=str, default=accelerator, help="Accelerator to use. Default is gpu. Alternative for local training: cpu.")
    parser.add_argument("--dataset", type=str, default=dataset, help="Dataset to use. Default is brats. Alternative: oasis.")
    parser.add_argument("--segs", action='store_true', default=segs, help="Do we load segmentations from the dataset.")
    parser.add_argument("--lms", action='store_true', default=lms, help="Do we load landmarks from the dataset.")
    parser.add_argument("--mask", action='store_true', default=mask, help="Do we load masks from the dataset.")
    parser.add_argument("--total_levels", type=int, default=total_levels)
    parser.add_argument("--latent_levels", type=int, default=latent_levels)
    parser.add_argument("--beta", type=float, default=beta)
    parser.add_argument("--batch_size", type=int, default=batch_size)
    parser.add_argument("--learning_rate", type=float, default=learning_rate)
    parser.add_argument("--recon_loss", nargs='+', default=recon_loss, help="Losses used in training. Default is mse. Options: mse, ncc, dice.")
    parser.add_argument("--dice_factor", type=int, default=50, help="Factor to scale dice up to MSE magnitude. Only relevant if dice is used as a training loss.")
    parser.add_argument("--gamma", type=float, default=gamma, help="Factor to scale ncc up to MSE magnitude.")
    parser.add_argument("--similarity_pyramid", action='store_true', default=similarity_pyramid, help="Whether to use a similarity pyramid or not.")
    parser.add_argument("--lambda", type=float, default=lamb, dest="lamb", help="Lambda of regularization. Setting to 0 equals no regularization.")
    parser.add_argument("--regularizer", type=str, default=regularizer, help="Regularizer to use. Default is L2. Alternatives: jdet.")
    parser.add_argument("--image_logging_frequency", type=int, default=image_logging_frequency)
    parser.add_argument("--feedback", nargs='+', default=feedback, help="Feedback connection between sampling layers. Default includes all options. Options: samples, velocity_field, individual_dfs, combined_dfs, final_dfs, transformed.")
    parser.add_argument("--df_resolution", type=str, default=df_resolution, help="Whether the dfs and thus transformed images are created at the resolution of 2x the sampling or at full resolution. Options: full_res, level_res.")
    parser.add_argument("--n0", type=int, default=32, help="Multiplier for the number of channels throughout the network")
    parser.add_argument("--ndims", type=int, default=ndims, help="Choose here if you want to work with volumes (3) or slices (2). Default is 3.")
    parser.add_argument("--interpatient", action='store_true', default=False, help="Whether to use the interpatient dataset or not. Only relevant for the BraTS dataset.")
    parser.add_argument("--nondiagonal", action='store_true', default=False, help="Whether to use the nondiagonal prior and respective KL loss or not.")
    parser.add_argument("--cp_depth", type=int, default=3, help="Depth of the control point layer. Default is 3.")
    
    args = parser.parse_args()

    main(args)