import random
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from lightning.pytorch import LightningDataModule
from torch import Generator, nn
from torch.utils.data import DataLoader, Dataset, IterableDataset

from ..misc.step_tracker import StepTracker
from . import DatasetCfg, get_dataset
from .types import DataShim, Stage
from .validation_wrapper import ValidationWrapper


def get_data_shim(encoder: nn.Module) -> DataShim:
    """获取用于修改批处理数据的函数。
    有时需要在数据加载器之外修改批处理数据，因为需要在GPU上进行计算，
    或者因为修改依赖于数据加载器之外的内容。
    """

    shims: list[DataShim] = []
    # 如果encoder对象有get_data_shim方法，那么调用该方法并将返回的函数添加到shims列表中。
    # 此时意味着encoder可能需要对数据进行特定的预处理。
    if hasattr(encoder, "get_data_shim"):
        shims.append(encoder.get_data_shim())

    # 这个内部函数遍历shims列表中的所有函数，对批处理数据batch依次进行处理。
    # 最终返回修改后的batch。
    def combined_shim(batch):
        for shim in shims:
            batch = shim(batch)
        return batch
    # get_data_shim函数返回combined_shim函数，供外部调用。
    return combined_shim


@dataclass
class DataLoaderStageCfg:
    batch_size: int
    num_workers: int
    persistent_workers: bool
    seed: int | None


@dataclass
class DataLoaderCfg:
    train: DataLoaderStageCfg
    test: DataLoaderStageCfg
    val: DataLoaderStageCfg


DatasetShim = Callable[[Dataset, Stage], Dataset]


def worker_init_fn(worker_id: int) -> None:
    random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))
    np.random.seed(int(torch.utils.data.get_worker_info().seed) % (2**32 - 1))


class DataModule(LightningDataModule):
    dataset_cfg: DatasetCfg
    data_loader_cfg: DataLoaderCfg
    step_tracker: StepTracker | None
    dataset_shim: DatasetShim
    global_rank: int

    def __init__(
        self,
        dataset_cfg: DatasetCfg,
        data_loader_cfg: DataLoaderCfg,
        step_tracker: StepTracker | None = None,
        dataset_shim: DatasetShim = lambda dataset, _: dataset,
        global_rank: int = 0,
    ) -> None:
        super().__init__()
        self.dataset_cfg = dataset_cfg
        self.data_loader_cfg = data_loader_cfg
        self.step_tracker = step_tracker
        self.dataset_shim = dataset_shim
        self.global_rank = global_rank

    def get_persistent(self, loader_cfg: DataLoaderStageCfg) -> bool | None:
        return None if loader_cfg.num_workers == 0 else loader_cfg.persistent_workers

    def get_generator(self, loader_cfg: DataLoaderStageCfg) -> torch.Generator | None:
        if loader_cfg.seed is None:
            return None
        generator = Generator()
        generator.manual_seed(loader_cfg.seed + self.global_rank)
        return generator

    def train_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "train", self.step_tracker)
        dataset = self.dataset_shim(dataset, "train")
        return DataLoader(
            dataset,
            self.data_loader_cfg.train.batch_size,
            shuffle=not isinstance(dataset, IterableDataset),
            num_workers=self.data_loader_cfg.train.num_workers,
            generator=self.get_generator(self.data_loader_cfg.train),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.train),
        )

    def val_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "val", self.step_tracker)
        dataset = self.dataset_shim(dataset, "val")
        return DataLoader(
            ValidationWrapper(dataset, 1),
            self.data_loader_cfg.val.batch_size,
            num_workers=self.data_loader_cfg.val.num_workers,
            generator=self.get_generator(self.data_loader_cfg.val),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.val),
        )

    def test_dataloader(self):
        dataset = get_dataset(self.dataset_cfg, "test", self.step_tracker)
        dataset = self.dataset_shim(dataset, "test")
        return DataLoader(
            dataset,
            self.data_loader_cfg.test.batch_size,
            num_workers=self.data_loader_cfg.test.num_workers,
            generator=self.get_generator(self.data_loader_cfg.test),
            worker_init_fn=worker_init_fn,
            persistent_workers=self.get_persistent(self.data_loader_cfg.test),
        )
