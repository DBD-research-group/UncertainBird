
from datasets import Audio
from datasets import load_dataset

from birdset.datamodule.components.transforms import BirdSetTransformsWrapper
from . import BaseDataModuleHF
from birdset.configs import DatasetConfig, LoadersConfig

class WABADDataModule(BaseDataModuleHF):
    def __init__(
        self,
        dataset: DatasetConfig = DatasetConfig(),
        loaders: LoadersConfig = LoadersConfig(),
    ):
        super().__init__(
            dataset=dataset,
            loaders=loaders,
        )

    @property
    def num_classes(self):
        return 1192

    def _load_data(self):
        # Override the base method to handle private dataset properly
        dataset = load_dataset(
            path=self.dataset_config.hf_path,
            name=self.dataset_config.hf_name,
            trust_remote_code=True,  # Needed for custom dataset scripts
        )
        return dataset

    def _preprocess_data(self, dataset):
        dataset = dataset.cast_column(
            column="audio",
            feature=Audio(
                sampling_rate=self.dataset_config.sample_rate,
                mono=True,
                decode=True,
            ),
        )
        dataset = dataset.select_columns(["audio", "labels"])
        return dataset