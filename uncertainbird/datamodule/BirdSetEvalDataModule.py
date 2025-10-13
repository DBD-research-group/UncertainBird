from datasets import DatasetDict

from birdset import utils
from birdset.datamodule import BirdSetDataModule

log = utils.get_pylogger(__name__)


class BirdSetEvalDataModule(BirdSetDataModule):

    def _preprocess_data(self, dataset):
        if self.dataset_config.task == "multiclass":
            # pick only train and test dataset
            dataset = DatasetDict(
                {split: dataset[split] for split in ["train", "test"]}
            )

            log.info("> Mapping data set.")
            dataset["train"] = dataset["train"].map(
                self.event_mapper,
                remove_columns=["audio"],
                batched=True,
                batch_size=300,
                load_from_cache_file=True,
                num_proc=self.dataset_config.n_workers,
                desc="Train event mapping",
            )

            if (
                self.dataset_config.class_weights_loss
                or self.dataset_config.class_weights_sampler
            ):
                self.num_train_labels = self._count_labels(
                    (dataset["train"]["ebird_code"])
                )

            if self.dataset_config.classlimit and not self.dataset_config.eventlimit:
                log.info(f">> Limiting classes to {self.dataset_config.classlimit}")
                dataset["train"] = self._limit_classes(
                    dataset=dataset["train"],
                    label_name="ebird_code",
                    limit=self.dataset_config.classlimit,
                )
            elif self.dataset_config.classlimit or self.dataset_config.eventlimit:
                log.info(
                    f">> Smart sampling to {self.dataset_config.classlimit=}, {self.dataset_config.eventlimit=}"
                )
                dataset["train"] = self._smart_sampling(
                    dataset=dataset["train"],
                    label_name="ebird_code",
                    class_limit=self.dataset_config.classlimit,
                    event_limit=self.dataset_config.eventlimit,
                )
            dataset = dataset.rename_column("ebird_code", "labels")

        elif self.dataset_config.task == "multilabel":
            # pick only train and test_5s dataset
            dataset = DatasetDict(
                {split: dataset[split] for split in ["train", "test_5s"]}
            )

            # log.info(">> Mapping train data.")
            # dataset["train"] = dataset["train"].map(
            #     self.event_mapper,
            #     remove_columns=["audio"],
            #     batched=True,
            #     batch_size=300,
            #     num_proc=self.dataset_config.n_workers,
            #     desc="Train event mapping",
            # )

            dataset = dataset.rename_column("ebird_code_multilabel", "labels")

            if self.dataset_config.classlimit or self.dataset_config.eventlimit:
                log.info(">> Smart Sampling")  #!TODO: implement custom caching?
                dataset["train"] = self._smart_sampling(
                    dataset=dataset["train"],
                    label_name="ebird_code",
                    class_limit=self.dataset_config.classlimit,
                    event_limit=self.dataset_config.eventlimit,
                )

            log.info(">> One-hot-encode classes")
            for split in ["test_5s"]:
                dataset[split] = dataset[split].map(
                    self._classes_one_hot,
                    batched=True,
                    batch_size=300,
                    load_from_cache_file=True,
                    num_proc=self.dataset_config.n_workers,
                    desc=f"One-hot-encoding {split} labels.",
                )

            if (
                self.dataset_config.class_weights_loss
                or self.dataset_config.class_weights_sampler
            ):
                self.num_train_labels = self._count_labels(
                    (dataset["train"]["ebird_code"])
                )

            dataset_test = dataset.pop("test_5s")
            dataset["test"] = dataset_test
        else:
            raise f"{self.dataset_config.task=} is not supported, choose (multilabel, multiclass)"

        for split in ["train", "test"]:
            dataset[split] = dataset[split].select_columns(
                ["filepath", "labels", "detected_events", "start_time", "end_time"]
            )

        return dataset
