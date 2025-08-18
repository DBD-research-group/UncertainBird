import os
import hydra
import lightning as L
from omegaconf import OmegaConf, open_dict
from omegaconf.errors import ConfigAttributeError
import json
from birdset import utils
import pyrootutils
from pathlib import Path
import torch.nn as nn
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

from mc_predictor import mc_predict


import torch


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

def log_dropout_layers(m):
    for name, module in m.named_modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout)):
            log.info(f"[DROPOUT] {name}: {module.__class__.__name__}(p={module.p})")

def describe_mcd_setup(model: nn.Module, max_name_len: int = 60):
    print("\n=== MC-Dropout Setup Summary ===")
    print(f"Hook flag: {getattr(model, '_eat_mc_flag', {'on': None})}")
    print("Hook count:", len(getattr(model, "_mcd_handles", [])))
    print("\n-- Hooked locations (by module name) --")
    found = False
    for name, mod in model.named_modules():
        tags = getattr(mod, "_mcd_tags", None)
        if tags:
            found = True
            tag_str = " | ".join(tags)
            nm = name if name else "<root>"
            if len(nm) > max_name_len:
                nm = "…" + nm[-(max_name_len-1):]
            print(f"{nm:<{max_name_len}}  ::  {mod.__class__.__name__}  -->  {tag_str}")
    if not found:
        print("(no hook tags found)")

    print("\n-- Native Dropout modules (state & p) --")
    any_do = False
    for name, mod in model.named_modules():
        if isinstance(mod, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            any_do = True
            nm = name if name else "<root>"
            print(f"{nm:<{max_name_len}}  ::  {mod.__class__.__name__}  training={mod.training}  p={mod.p}")
    if not any_do:
        print("(no native Dropout modules)")
    print("=== end ===\n")

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
    datamodule.prepare_data()  # has to be called before model for len_traindataset!

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

    log.info(model)

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
        log.info("Starting testing")

        # --- resolve checkpoint path exactly like before ---
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

        # --- branch: MC-Dropout vs normal test ---
        if cfg.get("MCTest", False):
            BaseCls = model.__class__
            best_base = BaseCls.load_from_checkpoint(ckpt_path, strict=True)

            log.info("MC-Dropout mode enabled → running predict loop with T stochastic passes (using in-memory weights)")
            # ensure import path to Wrappers is set (as you did before)
            target = getattr(best_base, "model", best_base)
            _eat_hooks = attach_eat_dropout_hooks_fine(
                target,
                p_conv_res=0.05,
                p_conv_down=0.05,
                p_project=0.05,
                p_token=0.05,
                p_head=0.05,
            )





            if hasattr(datamodule, "setup"):
                try:
                    datamodule.setup("test")   # prefer stage="test"
                except TypeError:
                    datamodule.setup() 
                    
            
            def set_dropout_p(model, p: float):
                model.eval()
                for m in model.modules():
                    if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
                        m.p = p
                        m.train()

            set_dropout_p(best_base, cfg.get("mcd_p"))
            set_eat_mc_mode(target, enabled=True, freeze_batchnorm=True)

            for m in best_base.modules():
                if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
                    print(f"[MC] {m.__class__.__name__} p={m.p} training={m.training}")



            # Now build the loader and pass it directly to predict()
            dl = datamodule.test_dataloader()
            describe_mcd_setup(best_base)
            mc_out = mc_predict(
                trainer=trainer,
                base_model=best_base,                 # your LightningModule or nn.Module
                dataloader=dl,                # or pass dataloader=...
                T=20,
                threshold=float(cfg.get("MCTest_threshold", 0.7)),
                task="multilabel",                    # or "multiclass"
                num_labels=21,                        # optional; inferred if possible
                compute_metrics=True,
                ckpt_path=None,                       # often None; you already loaded weights
             )
            # # mc_predict(
            #     trainer=trainer,
            #     base_model=best_base,          # <- use current model weights
            #     dataloader=dl,
            #     T=int(cfg.get("MCTest_T", 10)),
            #     threshold=float(cfg.get("MCTest_threshold", 0.7)),
            #     ckpt_path=ckpt_path,
            # )
            probs  = mc_out["p_mean"].detach().cpu()
            labels = mc_out["y"].detach().cpu().int()
            N, C = probs.shape

            auroc_macro = MultilabelAUROC(num_labels=C, average="macro")(probs, labels)
            map_micro   = MultilabelAveragePrecision(num_labels=C, average="micro")(probs, labels)

            def hit_at_k(p, y, k):
                topk = p.topk(k, dim=1).indices
                hits = y.gather(1, topk).max(1).values.float()
                return float(hits.mean())

            t1 = hit_at_k(probs, labels, 1)
            t3 = hit_at_k(probs, labels, 3)

            # cmAP5 (macro AP using only top-5 per sample)
            if C >= 5:
                top5 = probs.topk(5, dim=1).indices
                mask = torch.zeros_like(probs, dtype=torch.bool)
                mask.scatter_(1, top5, True)
                probs_top5 = torch.where(mask, probs, torch.zeros_like(probs))
            else:
                probs_top5 = probs
            cmAP5_macro = MultilabelAveragePrecision(num_labels=C, average="macro")(probs_top5, labels)

            # pcmAP (per-class AP averaged = macro mAP)
            map_per_class = MultilabelAveragePrecision(num_labels=C, average="none")(probs, labels)
            pcmAP_macro = float(map_per_class.mean())

            test_metrics = {
                "test/MultilabelAUROC": float(auroc_macro),
                "test/mAP":             float(map_micro),
                "test/T1Accuracy":      t1,
                "test/T3Accuracy":      t3,
                "test/cmAP5":           float(cmAP5_macro),
                "test/pcmAP":           pcmAP_macro,
            }

            for k, v in test_metrics.items():
                trainer.callback_metrics[k] = torch.tensor(v)

            trainer.callback_metrics.update({
                "mc/mean_var": torch.tensor(mc_out["p_var"].mean().item()),
                "mc/mean_prob": torch.tensor(mc_out["p_mean"].mean().item()),
            })




        else:
            trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)


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
