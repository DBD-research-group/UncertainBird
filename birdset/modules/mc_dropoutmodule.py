from dataclasses import asdict
import torch
import wandb
from typing import Callable, Literal, Type, Optional
from torch.nn import BCEWithLogitsLoss
from torch.nn.modules.loss import _Loss
from torch.optim import AdamW, Optimizer
from functools import partial

from .multilabel_module import MultilabelModule
from birdset.configs import (
    NetworkConfig,
    LRSchedulerConfig,
    MultilabelMetricsConfig,
    LoggingParamsConfig,
)


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
    def test_step(self, batch, batch_idx):

        testlossdict= []
        predsdict = []
        targetdict= []
        logitsdict= []
        for i in range (1):

            test_loss, preds, targets, logits = self.model_step(batch, batch_idx, return_logits=True)
            testlossdict.append(test_loss.detach().cpu())
            predsdict.append(preds.detach().cpu())
            targetdict.append(targets.detach().cpu())
            logitsdict.append(logits.detach().cpu())

        # save targets and predictions for test_epoch_end
        self.test_targets.append(targets.detach().cpu())
        self.test_preds.append(preds.detach().cpu())

        self.log(
            f"test/{self.loss.__class__.__name__}",
            test_loss,
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

        return {"loss": test_loss, "preds": preds, "targets": targets}
    