from dataclasses import asdict
import torch
import wandb
from typing import Callable, Literal, Type, Optional
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from functools import partial

from .multilabel_module import MultilabelModule
from projects.uncertainbird.configs.experiment.Wrappers.resolve_models import resolve_soundnet_eat
from birdset.configs import (
    NetworkConfig,
    LRSchedulerConfig,
    MultilabelMetricsConfig,
    LoggingParamsConfig,
)
from projects.uncertainbird.configs.experiment.Wrappers.eat_dropout_hooks import attach_eat_dropout_hooks_fine, set_eat_mc_mode, describe_eat_setup

class MCDropoutModule(MultilabelModule):
    """
    MultilabelModule is a PyTorch Lightning module for multilabel classification tasks.

    Attributes:
        prediction_table (bool): Whether to create a prediction table. Defaults to False.
    """

    def __init__(
        self,
        network: NetworkConfig = NetworkConfig(),
        output_activation: Callable[[torch.Tensor], torch.Tensor] = torch.sigmoid,
        loss: _Loss = BCEWithLogitsLoss(),
        optimizer: partial[Type[Optimizer]] = partial(
            AdamW,
            lr=1e-5,
            weight_decay=0.01,
        ),
        lr_scheduler: Optional[LRSchedulerConfig] = LRSchedulerConfig(),
        metrics: MultilabelMetricsConfig = MultilabelMetricsConfig(),
        logging_params: LoggingParamsConfig = LoggingParamsConfig(),
        num_epochs: int = 50,
        len_trainset: int = 13878,  # set to property from datamodule
        batch_size: int = 32,
        task: Literal["multiclass", "multilabel"] = "multilabel",
        num_gpus: int = 1,
        prediction_table: bool = False,
        pretrain_info=None,
        mask_logits: bool = True,
        
    ):

        self.prediction_table = prediction_table

        super().__init__(
            network=network,
            output_activation=output_activation,
            loss=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            metrics=metrics,
            logging_params=logging_params,
            num_epochs=num_epochs,
            len_trainset=len_trainset,
            task=task,
            batch_size=batch_size,
            num_gpus=num_gpus,
            pretrain_info=pretrain_info,
            mask_logits=mask_logits,
            
        )
    def _resolve_hf_convnext(self):
        m = self.model
        # case 1: already the HF model
        if hasattr(m, "convnext"):
            return m
        # case 2: wrapper with .model -> HF
        inner = getattr(m, "model", None)
        if inner is not None and hasattr(inner, "convnext"):
            return inner
        # fallback: search children
        for mod in m.modules():
            if hasattr(mod, "convnext"):
                return mod
        raise RuntimeError("Could not find HF ConvNext model (no .convnext found)")
    def test_step(self, batch, batch_idx):

        T = getattr(self, "T", 10) 
        logits_list, loss_list = [], []
        for i in range (T):

            loss, preds, targets, logits = self.model_step(batch, batch_idx, return_logits=True)
            logits_list.append(logits)        # [B, C] logits for this pass
            loss_list.append(loss)            # scalar (tensor)
            targets_ref = targets 

        logits_T   = torch.stack(logits_list, dim=0)
        logit_mean = logits_T.mean(dim=0)                     # [B, C]
        logit_var  = logits_T.var(dim=0, unbiased=False)

        
        p_mean = torch.sigmoid(logit_mean)                # [B, C]
        probs_T = torch.sigmoid(logits_T)   
        loss_mc   = torch.stack(loss_list).mean()              # [T, B, C]
        p_var  = probs_T.var(dim=0, unbiased=False)       # [B, C]
        y_hat  = (p_mean > getattr(self, "threshold", 0.5)).int()


        # save targets and predictions for test_epoch_end
        self.test_targets.append(targets_ref.detach().cpu())
        self.test_preds.append(p_mean.detach().cpu())

        self.log(
            f"test/{self.loss.__class__.__name__}",
            loss_mc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.test_metric(preds, targets.int())
        self.log(
            f"test/{self.test_metric.__class__.__name__}",
            self.test_metric,
            **asdict(self.logging_params),
        )
        self.test_add_metrics(preds, targets.int())
        self.log_dict(self.test_add_metrics, **asdict(self.logging_params))

        return {"loss": loss_mc, "preds": preds, "targets": targets}


    def on_test_epoch_start(self):
        self.T =10
        hf = resolve_soundnet_eat(self.model)
        self.p_conv_res = 0.0
        self.p_conv_down = 0.00
        self.p_project = 0.0
        self.p_token = 0.0
        self.p_head = 0.04
        self.use_hooks= True


        self.print("starting test epoch")
        
        attach_eat_dropout_hooks_fine(hf,p_conv_res=self.p_conv_res, p_conv_down=self.p_conv_down, p_project=self.p_project, p_token=self.p_token,p_head=self.p_head)
        set_eat_mc_mode(hf, enabled=(self.T > 1 and self.use_hooks))
        # self.print(f"[MC] hooks {'ENABLED' if (self.T > 1 and self.use_hooks) else 'disabled'} (T={self.T})")

        describe_eat_setup(hf)
        
        return super().on_test_epoch_start()