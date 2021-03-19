from typing import Optional, Sequence

import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from torchmeta.transforms import ClassSplitter
from torchmeta.utils.data import BatchMetaDataLoader
from torchvision.transforms import Compose


class CustomBatchMetaDataLoader(BatchMetaDataLoader):
    def __init__(self, dataset, num_epochs=1, batch_size=1, shuffle=True,
                 sampler=None, num_workers=0, pin_memory=False,
                 drop_last=False, timeout=0, worker_init_fn=None):
        self.num_epochs = num_epochs
        super(CustomBatchMetaDataLoader, self).__init__(dataset,
                                                        batch_size=batch_size,
                                                        shuffle=shuffle,
                                                        sampler=sampler,
                                                        num_workers=num_workers,
                                                        pin_memory=pin_memory,
                                                        drop_last=drop_last,
                                                        timeout=timeout,
                                                        worker_init_fn=worker_init_fn)

    def __len__(self):
        return 1


class MetaDataModule(pl.LightningDataModule):
    def __init__(
            self,
            nway: int,
            datasets: DictConfig,
            num_workers: DictConfig,
            kshot: DictConfig,
            transforms: DictConfig,
            target_transform: DictConfig,
            class_augmentations: DictConfig,
            batch_size: DictConfig,
            num_meta_batches: DictConfig,
            num_inner_steps: DictConfig,
            cfg: DictConfig,
    ):
        super().__init__()
        self.cfg = cfg

        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.kshot = kshot
        self.num_meta_batches = num_meta_batches
        self.nway = nway
        self.random_seed = cfg.train.random_seed
        self.transforms = transforms
        self.class_augmentations = class_augmentations
        self.target_transform = target_transform
        self.num_inner_steps = num_inner_steps

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

    def prepare_data(self) -> None:
        # download only
        pass

    def setup(self, stage: Optional[str] = None):
        transform_compose = Compose([instantiate(transform) for
                                     transform in self.transforms])
        target_transform = instantiate(self.target_transform,
                                       num_classes=self.nway)
        class_augmentations = [
            instantiate(augmentation)
            for augmentation in
            self.class_augmentations]
        # Here you should instantiate your datasets, you may also split the train into train and validation if needed.
        if stage is None or stage == "fit":
            self.train_dataset = instantiate(
                self.datasets.train,
                num_classes_per_task=self.nway,
                transform=transform_compose,
                target_transform=target_transform,
                class_augmentations=class_augmentations)
            self.train_dataset.seed(self.random_seed)
            self.train_dataset = ClassSplitter(self.train_dataset,
                                               shuffle=True,
                                               random_state_seed=self.random_seed,
                                               num_support_per_class=self.kshot.support,
                                               num_query_per_class=self.kshot.query)
            self.val_dataset = instantiate(self.datasets.val,
                                           num_classes_per_task=self.nway,
                                           transform=transform_compose,
                                           target_transform=target_transform,
                                           class_augmentations=class_augmentations)
            self.val_dataset.seed(self.random_seed)
            self.val_dataset = ClassSplitter(self.val_dataset,
                                             shuffle=True,
                                             random_state_seed=self.random_seed,
                                             num_support_per_class=self.kshot.support,
                                             num_query_per_class=self.kshot.query)

        if stage is None or stage == "test":
            self.test_dataset = instantiate(self.datasets.test,
                                            num_classes_per_task=self.nway,
                                            transform=transform_compose,
                                            target_transform=target_transform,
                                            class_augmentations=class_augmentations)
            self.test_dataset.seed(self.random_seed)
            self.test_dataset = ClassSplitter(self.test_dataset,
                                              shuffle=True,
                                              random_state_seed=self.random_seed,
                                              num_support_per_class=self.kshot.support,
                                              num_query_per_class=self.kshot.query)

    def train_dataloader(self) -> DataLoader:
        return CustomBatchMetaDataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
        )

    def val_dataloader(self) -> DataLoader:
        return CustomBatchMetaDataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_size.val,
            num_workers=self.num_workers.val,
        )

    def test_dataloader(self) -> DataLoader:
        return CustomBatchMetaDataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_size.test,
            num_workers=self.num_workers.test,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets}, "
            f"{self.num_workers}, "
            f"{self.batch_size})"
        )
