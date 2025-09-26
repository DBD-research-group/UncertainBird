from datasets import load_dataset, Audio

from birdset.datamodule.wabad_datamodule import WABADDataModule
from birdset.configs import DatasetConfig, LoadersConfig

def main():
    # Setup dataset config (optional: override defaults)
    dataset_cfg = DatasetConfig()
    dataset_cfg.hf_path = "DBD-research-group/WABAD"   # Hugging Face dataset
    dataset_cfg.hf_name = "BAM"
    dataset_cfg.sample_rate = 48000
    dataset_cfg.val_split = None

    # Load datamodule
    dm = WABADDataModule(
        dataset=dataset_cfg,
        loaders=LoadersConfig(),
    )

    # Prepare data (downloads from Hugging Face if needed)
    dm.prepare_data()

    # print(dm.data_path)
    print(dm.num_classes)
    print(dm.len_trainset)

    data = dm._load_data()
    print(data["test"][0])


if __name__ == "__main__":
    main()
