import os
import pickle
import torch
import numpy as np
from pathlib import Path
from lightning.pytorch.callbacks import Callback
from typing import Any, Dict, List, Optional
from birdset.utils import pylogger

log = pylogger.get_pylogger(__name__)


class DumpPredictionsCallback(Callback):
    """
    A PyTorch Lightning callback that saves all predictions and targets from the test run.

    This callback collects predictions and targets during test steps and saves them
    to disk at the end of the test epoch for further analysis.

    Args:
        save_dir (str): Directory where to save the predictions and targets.
        filename_prefix (str): Prefix for the saved files. Defaults to "test_predictions".
        save_format (str): Format to save the data. Options: "pickle", "numpy". Defaults to "pickle".
        save_logits (bool): Whether to save raw logits or apply output activation. Defaults to True.
    """

    def __init__(
        self,
        save_dir: str = "predictions",
        filename_prefix: str = "test_predictions",
        save_format: str = "pickle",
    ):
        super().__init__()
        self.save_dir = Path(save_dir)
        self.filename_prefix = filename_prefix
        self.save_format = save_format

        # Ensure save directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)

        # Storage for predictions and targets
        self.test_predictions: List[torch.Tensor] = []
        self.test_targets: List[torch.Tensor] = []
        self.test_metadata: List[Dict[str, Any]] = []
        log.info(
            f"Initialized DumpPredictionsCallback with save_dir: {self.save_dir}, filename_prefix: {self.filename_prefix}, save_format: {self.save_format}, save_logits: {self.save_logits}"
        )

    def on_test_start(self, trainer, pl_module):
        """Called at the beginning of testing."""
        print(f"Starting prediction collection. Saving to: {self.save_dir}")
        # Clear any previous data
        self.test_predictions = []
        self.test_targets = []
        self.test_metadata = []

    def on_test_batch_end(
        self,
        trainer,
        pl_module,
        outputs: Dict[str, Any],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int = 0,
    ):
        """Called after each test batch."""
        if outputs is not None:
            # Extract predictions and targets from outputs
            preds = outputs.get("preds", None)
            targets = outputs.get("targets", None)

            if preds is not None and targets is not None:
                # Move to CPU and detach from computation graph
                preds_cpu = preds.detach().cpu()
                targets_cpu = targets.detach().cpu()

                self.test_predictions.append(preds_cpu)
                self.test_targets.append(targets_cpu)

                # Store metadata about this batch
                metadata = {
                    "batch_idx": batch_idx,
                    "dataloader_idx": dataloader_idx,
                    "batch_size": preds.shape[0],
                    "prediction_shape": list(preds.shape),
                    "target_shape": list(targets.shape),
                }
                self.test_metadata.append(metadata)

    def on_test_epoch_end(self, trainer, pl_module):
        """Called at the end of the test epoch to save all collected data."""
        if not self.test_predictions or not self.test_targets:
            log.warning("Warning: No predictions or targets collected during testing.")
            return

        log.info(f"Saving {len(self.test_predictions)} batches of predictions...")

        # Concatenate all predictions and targets
        all_predictions = torch.cat(self.test_predictions, dim=0)
        all_targets = torch.cat(self.test_targets, dim=0)

        # Create data dictionary
        data = {
            "predictions": all_predictions,
            "targets": all_targets,
            "metadata": {
                "total_samples": all_predictions.shape[0],
                "prediction_shape": list(all_predictions.shape),
                "target_shape": list(all_targets.shape),
                "num_batches": len(self.test_predictions),
                "batch_metadata": self.test_metadata,
                "save_logits": self.save_logits,
                "model_info": {
                    "class_name": pl_module.__class__.__name__,
                    "task": getattr(pl_module, "task", "unknown"),
                    "num_classes": getattr(pl_module, "num_classes", None),
                },
            },
        }

        # Add experiment info if available
        if hasattr(trainer, "logger") and trainer.logger is not None:
            if hasattr(trainer.logger, "experiment"):
                experiment = trainer.logger.experiment
                if hasattr(experiment, "name"):
                    data["metadata"]["experiment_name"] = experiment.name
                if hasattr(experiment, "id"):
                    data["metadata"]["experiment_id"] = experiment.id

        # Generate filename with timestamp
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.save_format == "pickle":
            filename = f"{self.filename_prefix}_{timestamp}.pkl"
            filepath = self.save_dir / filename

            with open(filepath, "wb") as f:
                pickle.dump(data, f)

        elif self.save_format == "numpy":
            # Save predictions and targets as separate numpy files
            pred_filename = f"{self.filename_prefix}_predictions_{timestamp}.npy"
            target_filename = f"{self.filename_prefix}_targets_{timestamp}.npy"
            metadata_filename = f"{self.filename_prefix}_metadata_{timestamp}.pkl"

            pred_filepath = self.save_dir / pred_filename
            target_filepath = self.save_dir / target_filename
            metadata_filepath = self.save_dir / metadata_filename

            np.save(pred_filepath, all_predictions.numpy())
            np.save(target_filepath, all_targets.numpy())

            with open(metadata_filepath, "wb") as f:
                pickle.dump(data["metadata"], f)

            print(f"Saved predictions to: {pred_filepath}")
            print(f"Saved targets to: {target_filepath}")
            print(f"Saved metadata to: {metadata_filepath}")

        else:
            raise ValueError(f"Unsupported save format: {self.save_format}")

        if self.save_format == "pickle":
            print(f"Saved predictions and targets to: {filepath}")

        # Clear data to free memory
        self.test_predictions = []
        self.test_targets = []
        self.test_metadata = []

        print(f"Successfully saved {data['metadata']['total_samples']} test samples.")
