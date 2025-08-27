import os
import hydra
import lightning as L
from omegaconf import OmegaConf, open_dict
from omegaconf.errors import ConfigAttributeError
import json
from birdset import utils
import pyrootutils
from pathlib import Path
import sys
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision


# /workspace/birdset/train.py  -> parent twice = /workspace
WORKSPACE = Path(__file__).resolve().parents[1]

# Path where mc_predictor.py lives
WRAPPER_DIR = WORKSPACE / "projects" / "uncertainbird" / "configs" / "experiment" / "Wrappers"

# Make sure Python can find that folder (robust even when Hydra changes CWD)
sys.path.insert(0, str(WRAPPER_DIR))

# Now import your class/function
from eat_dropout_hooks import attach_eat_dropout_hooks_fine, set_eat_mc_mode
from convnext_dropout_hooks import attach_convnext_eat_hooks, set_convnext_eat_mc_mode, describe_convnext_eat_setup

from mc_predictor import mc_predict
from mc_Dropout import predict_withmc

log = utils.get_pylogger(__name__)

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git"],
    pythonpath=True,
    dotenv=True,
)

_HYDRA_PARAMS = {
    "version_base": None,
    "config_path": str(root / "configs"),
    "config_name": "train.yaml",
}
def resolve_hf_convnext(obj):
    cur = obj
    for _ in range(3):  # unwrap up to 3 times
        if hasattr(cur, "convnext") and hasattr(cur, "classifier"):
            return cur  # ConvNextForImageClassification
        cur = getattr(cur, "model", cur)
    return cur


# @utils.register_custom_resolvers(**_HYDRA_PARAMS)
@hydra.main(**_HYDRA_PARAMS)
def train(cfg):
    log.info("Using config: \n%s", OmegaConf.to_yaml(cfg))

    log.info(f"Dataset path: <{os.path.abspath(cfg.paths.dataset_path)}>")
    os.makedirs(cfg.paths.dataset_path, exist_ok=True)

    log.info(f"Log path: <{os.path.abspath(cfg.paths.log_dir)}>")
    os.makedirs(cfg.paths.log_dir, exist_ok=True)

    log.info(f"Root Dir:<{os.path.abspath(cfg.paths.log_dir)}>")
    log.info(f"Work Dir:<{os.path.abspath(cfg.paths.work_dir)}>")
    log.info(f"Output Dir:<{os.path.abspath(cfg.paths.output_dir)}>")
    log.info(f"Background Dir:<{os.path.abspath(cfg.paths.background_path)}>")

    log.info(f"Seed everything with <{cfg.seed}>")
    L.seed_everything(cfg.seed)
    # log.info(f"Instantiate logger {[loggers for loggers in cfg['logger']]}")

    # Setup data
    log.info(f"Instantiate datamodule <{cfg.datamodule._target_}>")
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    datamodule.prepare_data()
  # has to be called before model for len_traindataset!

    # Setup logger
    log.info(f"Instantiate logger")
    logger = utils.instantiate_loggers(cfg.get("logger"))
    # override standard TF logger to handle rare logger error
    # logger.append(utils.TBLogger(Path(cfg.paths.log_dir)))

    # Setup callbacks
    log.info(f"Instantiate callbacks")
    callbacks = utils.instantiate_callbacks(cfg["callbacks"])

    # Training
    log.info(f"Instantiate trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=logger)

    pretrain_info = None
    try:
        pretrain_info = cfg.module.network.model.pretrain_info
    except ConfigAttributeError:
        log.info("pretrain_info does not exist using None instead")

    # Setup model
    log.info(f"Instantiate model <{cfg.module.network.model._target_}>")
    with open_dict(cfg):
        cfg.module.metrics["num_labels"] = datamodule.num_classes
        cfg.module.network.model["num_classes"] = (
            datamodule.num_classes
        )  # TODO not the correct classes when masking in valid/test only

    model = hydra.utils.instantiate(
        cfg.module,
        num_epochs=cfg.trainer.max_epochs,
        len_trainset=datamodule.len_trainset,
        batch_size=datamodule.loaders_config.train.batch_size,
        pretrain_info=pretrain_info,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    log.info("Logging Hyperparams")
    utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info(f"Starting training")
        ckpt = cfg.get("ckpt_path")
        if ckpt:
            log.info(f"Resume training from checkpoint {ckpt}")
        else:
            log.info("No checkpoint found. Training from scratch!")

        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info(f"Starting testing")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            if cfg.get("ckpt_path") == "":
                log.warning("No ckpt saved or found. Using current weights for testing")
                ckpt_path = None
            else:
                ckpt_path = cfg.get("ckpt_path")
                log.info(
                    f"The best checkpoint for {cfg.callbacks.model_checkpoint.monitor}"
                    f" is {trainer.checkpoint_callback.best_model_score}"
                    f" and saved in {ckpt_path}"
                )
        else:
            log.info(
                f"The best checkpoint for {cfg.callbacks.model_checkpoint.monitor}"
                f" is {trainer.checkpoint_callback.best_model_score}"
                f" and saved in {ckpt_path}"
            )
        if cfg.get("uq_method", None)!= None:
            match cfg.get("uq_method"):
                case "MCDropout":
                    log.info("MC-Dropout Testing → (using in-memory weights)")
                    base_model = model
                    #describe_convnext_eat_setup(target)
                    output_dir = cfg.paths.output_dir
                    mc_out = predict_withmc(
                        trainer=trainer,
                        base_model=base_model,                 # your LightningModule or nn.Module
                        datamodule=datamodule,                # or pass dataloader=...
                        T=1,
                        threshold=float(cfg.get("MCTest_threshold", 0.5)),
                        task="multilabel",                    # or "multiclass"
                        num_labels=21,                        # optional; inferred if possible
                        compute_metrics=True,
                        ckpt_path=None,                       # often None; you already loaded weights
                        plot_path = output_dir,
                    )


                    pass
                case "Ensemble":
                    pass
            
        else:
            log.info("Testing Without UQ Method → (using in-memory weights)")

            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
            # from pathlib import Path

            # local_ckpt = Path(cfg.paths.output_dir) / "model-local.ckpt"
            # trainer.save_checkpoint(str(local_ckpt))
            # print(f"Saved to {local_ckpt}")

    test_metrics = trainer.callback_metrics

    if cfg.get("save_state_dict"):
        log.info(f"Saving state dicts")
        utils.save_state_dicts(
            trainer=trainer,
            model=model,
            dirname=cfg.paths.output_dir,
            **cfg.extras.state_dict_saving_params,
        )

    if cfg.get("dump_metrics"):
        log.info(f"Dumping final metrics locally to {cfg.paths.output_dir}")
        metric_dict = {**train_metrics, **test_metrics}

        metric_dict = [
            {"name": k, "value": v.item() if hasattr(v, "item") else v}
            for k, v in metric_dict.items()
        ]

        file_path = os.path.join(cfg.paths.output_dir, "finalmetrics.json")
        with open(file_path, "w") as json_file:
            json.dump(metric_dict, json_file)

    utils.close_loggers()


if __name__ == "__main__":
    train()