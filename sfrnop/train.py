from sfrnop.data import define_dataloader
import torchinfo
import hydra
from omegaconf import DictConfig, OmegaConf

from lightning import Trainer, seed_everything
from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger, CSVLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from dotenv import load_dotenv
import logging
from pathlib import Path 
import coolname

from sfrnop.models.setonet import get_batch_mask
import numpy as np 

def nmse_tot(input, target):
    output = 10*np.log10(np.linalg.norm(input - target)**2/(np.linalg.norm(target)**2))
    return output

log = logging.getLogger(__name__)

def get_pretrained_model_checkpoint(pretrained_model: str | Path, work_folder: str | Path, ckpt_name: str = "best") -> Path:
    """
    Get pretrained model checkpoint path.

    Args:
        pretrained_model (str | Path): Name of the pretrained model or path to the model folder.
        work_folder (str | Path): Path to the work folder where pretrained models are stored.
        ckpt_name (str, optional): Name of the checkpoint file (without extension). Defaults to "best".

    Returns:
        Path: Path to the pretrained model checkpoint.
    """

    if Path(pretrained_model).is_file():
        return Path(pretrained_model)

    else:
        return Path(work_folder) / "models" / pretrained_model / "checkpoints" / f"{ckpt_name}.ckpt"
    
@hydra.main(version_base=None, config_path="configs", config_name="base")
def train(config: DictConfig):
    log.info("Training")

    training_config = config.training
    
    if training_config.alias is None:
        alias = coolname.generate_slug(2)
        training_config.alias = alias
    else:
        alias = training_config.alias
    
    log.info(f"Training alias: {alias}")

    output_folder = training_config.output_folder
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    config_path = Path(output_folder) / "config.yaml"
    if config_path.exists():
        if not training_config.get("overwrite", False):
            raise FileExistsError(f"Configuration file already exists at {config_path}. Please set 'training_config.overwrite=True' to overwrite.")
        else:
            log.warning(f"Overwriting existing configuration file at {config_path}.")
    OmegaConf.save(config, config_path, resolve=True)

    log.info(f"Training configuration\n{OmegaConf.to_yaml(config, resolve=True)}")

    seed_everything(training_config.seed, workers=True)
    log.info(f"Seed: {training_config.seed}")

    # Instantiate datasets with original code from Palette model
    train_dataloader, val_dataloader = define_dataloader(log, training_config, 'train')

    training_config.trainer["log_every_n_steps"] = min(training_config.trainer.get("log_every_n_steps", 50), len(train_dataloader))

    model: LightningModule = hydra.utils.instantiate(config.model)

    torchinfo.summary(model)
    
    loggers = []

    try:
        wandb_tags = training_config.get("tags", None)
        loggers.append(
            WandbLogger(
                name=alias,
                tags=wandb_tags
            )
        )
        log.info("Wandb logger initialized.")
    except ModuleNotFoundError:
        log.warning("Wandb is not installed. Skipping Wandb logger.")
    
    if len(loggers) == 0:
        log.info("No loggers found. Adding CSV logger.")
        loggers.append(
            CSVLogger(
                save_dir=output_folder,
                name=alias,
            )
        )
    
    hparams = OmegaConf.to_container(config, resolve=True)
    for logger in loggers:
        logger.log_hyperparams(hparams)

    
    early_stop = EarlyStopping(**training_config.callbacks.get("early_stopping", {}))

    model_checkpoint_params = OmegaConf.to_container(
        training_config.callbacks.get("model_checkpoint", {}), resolve=True
    )
    model_checkpoint_params.setdefault("dirpath", Path(output_folder) / "checkpoints")
    model_checkpoint = ModelCheckpoint(**model_checkpoint_params)

    trainer = Trainer(
        **training_config.get("trainer", {}),
        logger=loggers,
        callbacks=[early_stop, model_checkpoint],
    )

    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
        **training_config.get("fit", {})
    )

    log.info(f"Best model checkpoint at: {model_checkpoint.best_model_path}")

    return trainer

if __name__ == "__main__":
    load_dotenv()
    train()