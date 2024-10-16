import math
from typing import Any, Dict, Optional, Tuple

import hydra
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.optim
import wandb
from omegaconf import DictConfig
from torch import nn
from torch.utils.data import DataLoader, Dataset


class LRCosineScheduler(torch.optim.lr_scheduler.LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        cycle_steps: int,
        cycle_mult: float = 1.0,
        max_lr: float = 1e-3,
        min_lr: float = 1e-4,
        gamma: float = 1.0,
        last_epoch: int = -1,
        verbose: str = "deprecated",
    ) -> None:
        assert warmup_steps < cycle_steps
        self.warmup_steps = warmup_steps
        self.cycle_steps = cycle_steps
        self.cycle_mult = cycle_mult
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.gamma = gamma
        self.verbose = verbose

        self.cur_cycle_steps = self.cycle_steps
        self.last_cycle_steps = -1
        super().__init__(optimizer, last_epoch, verbose)
        self._init_lr()

    def _init_lr(self) -> None:
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.min_lr

    def get_lr(self) -> list[float]:
        if self.last_cycle_steps < self.warmup_steps:
            return [
                (self.max_lr - self.min_lr) / self.warmup_steps * self.last_cycle_steps
                + self.min_lr
                for _ in self.optimizer.param_groups
            ]
        else:
            decay = (self.last_cycle_steps - self.warmup_steps) / (
                self.cur_cycle_steps - self.warmup_steps
            )

            coeff = 0.5 * (1.0 + math.cos(math.pi * min(decay, 1.0)))
            return [
                self.min_lr + coeff * (self.max_lr - self.min_lr)
                for _ in self.optimizer.param_groups
            ]

    def step(self, epoch: Optional[int] = None) -> None:
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch

        if self.last_cycle_steps >= self.cur_cycle_steps:
            self.max_lr = (self.max_lr - self.min_lr) * self.gamma + self.min_lr
            self.cur_cycle_steps = int(self.cur_cycle_steps * self.cycle_mult)
            self.last_cycle_steps = 1
        else:
            self.last_cycle_steps += 1
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr


class ImplicitDataset(Dataset):
    def __init__(
        self,
        users: torch.Tensor,
        items: torch.Tensor,
        targets: torch.Tensor = None,
        weights: torch.Tensor = None,
        *args,
        **kwargs,
    ):
        self.users = users
        self.items = items
        self.targets = targets
        self.weights = weights

    def __len__(self) -> int:
        return self.users.size(0)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        user = self.users[idx]
        item = self.items[idx]
        if self.targets is not None and self.weights is not None:
            target = self.targets[idx]
            weight = self.weights[idx]
            return user, item, target, weight
        else:
            return user, item


class DataModule(pl.LightningDataModule):
    def __init__(
        self,
        users_path: str = "https://storage.yandexcloud.net/ds-ods/files/files/c1992ccf/users_meta.parquet",
        items_path: str = "https://storage.yandexcloud.net/ds-ods/files/files/13b479ed/items_meta.parquet",
        train_path: str = "https://storage.yandexcloud.net/ds-ods/files/data/docs/competitions/VKRecsysChallenge2024/dataset/train_interactions.parquet",
        test_path: str = "https://storage.yandexcloud.net/ds-ods/files/files/0235d298/test_pairs.csv",
        # weights: Tuple[float, float] = (0.5, 10.0, 20.0),
        weights: Tuple[float, float] = (0.35, 7.15, 850.0),
        val_size: float = 0.1,
        batch_size: int = 128,
        num_workers: int = 2,
        pin_memory: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(logger=False)
        self.users_path = users_path
        self.items_path = items_path
        self.train_path = train_path
        self.test_path = test_path

        self.weights = weights
        self.batch_size = batch_size
        self.val_size = val_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if self.trainer is not None:
            if self.batch_size % self.trainer.world_size != 0:
                raise RuntimeError(
                    f"Batch size ({self.batch_size}) is not divisible by the number of devices ({self.trainer.world_size})."
                )
            self.batch_size_per_device = self.batch_size // self.trainer.world_size
        else:
            self.batch_size_per_device = self.batch_size
        if not self.train_dataset and not self.val_dataset and not self.test_dataset:
            users = pd.read_parquet(self.users_path)
            items = pd.read_parquet(self.items_path)
            interactions = pd.read_parquet(self.train_path)

            items = items.drop(columns=["embeddings"])
            interactions["target"] = interactions.like.astype(
                np.int8
            ) - interactions.dislike.astype(np.int8)
            interactions = interactions.drop(
                columns=["timespent", "like", "dislike", "share", "bookmarks"]
            )
            interactions = interactions.merge(users, on="user_id", how="left")
            interactions = interactions.merge(items, on="item_id", how="left")
            train_idx = int((1.0 - self.val_size) * interactions.shape[0])
            train_users = torch.from_numpy(
                interactions.iloc[:train_idx, [0, 3, 4]].values
            )
            train_items = torch.from_numpy(
                interactions.iloc[:train_idx, [1, 5, 6]].values
            )
            train_targets = torch.from_numpy(interactions.iloc[:train_idx, 2].values)
            train_weights = torch.full_like(
                train_targets, fill_value=self.weights[0], dtype=torch.float32
            )
            train_weights[train_targets == 1] = self.weights[1]
            train_weights[train_targets == -1] = self.weights[2]
            val_users = torch.from_numpy(
                interactions.iloc[train_idx:, [0, 3, 4]].values
            )
            val_items = torch.from_numpy(
                interactions.iloc[train_idx:, [1, 5, 6]].values
            )
            val_targets = torch.from_numpy(interactions.iloc[train_idx:, 2].values)
            val_weights = torch.full_like(
                val_targets, fill_value=self.weights[0], dtype=torch.float32
            )
            val_weights[val_targets == 1] = self.weights[1]
            val_weights[val_targets == -1] = self.weights[2]
            test = pd.read_csv(self.test_path)
            test = test.merge(users, on="user_id", how="left")
            test = test.merge(items, on="item_id", how="left")
            test_users = torch.from_numpy(test.iloc[:, [0, 2, 3]].values)
            test_items = torch.from_numpy(test.iloc[:, [1, 4, 5]].values)
            self.train_dataset = ImplicitDataset(
                train_users, train_items, train_targets, train_weights
            )
            self.val_dataset = ImplicitDataset(
                val_users, val_items, val_targets, val_weights
            )
            self.test_dataset = ImplicitDataset(test_users, test_items)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


class RecsysModel(nn.Module):
    def __init__(
        self,
        user_size: int,
        item_size: int,
        source_size: int,
        pretrained_emb: torch.Tensor,
        user_emb_size: int = 64,
        item_emb_size: int = 64,
        source_emb_size: int = 8,
        gender_emb_size: int = 8,
        duration_emb_size: int = 8,
        age_emb_size: int = 8,
        dropout: float = 0.4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.user_emb = nn.Embedding(user_size, user_emb_size)
        self.item_emb = nn.Embedding(item_size, item_emb_size)
        self.source_emb = nn.Embedding(source_size, source_emb_size)
        self.gender_emb = nn.Embedding(3, gender_emb_size)
        self.duration_emb = nn.Linear(1, duration_emb_size)
        self.age_emb = nn.Linear(1, age_emb_size)
        self.register_buffer("pretrained_emb", pretrained_emb)
        emb_size = (
            user_emb_size
            + item_emb_size
            + source_emb_size
            + gender_emb_size
            + duration_emb_size
            + age_emb_size
            + pretrained_emb.size(-1)
        )
        self.main = nn.Sequential(
            nn.Dropout(dropout),
            nn.LayerNorm(emb_size),
            nn.Linear(emb_size, 256),
            nn.SiLU(True),
            nn.Dropout(dropout),
            nn.LayerNorm(256),
            nn.Linear(256, 128),
            nn.SiLU(True),
            nn.Dropout(dropout),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.SiLU(True),
            nn.Dropout(dropout),
            nn.LayerNorm(64),
            nn.Linear(64, 1),
        )

    def forward(self, user: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        user_emb = self.user_emb(user[:, 0].long())
        gender_emb = self.gender_emb(user[:, 1].long())
        age_emb = self.age_emb(user[:, 2].unsqueeze(-1).float())

        item_emb = self.item_emb(items[:, 0].long())
        item_pre_emb = self.pretrained_emb[items[:, 0].long()]
        source_emb = self.source_emb(items[:, 1].long())
        duration_emb = self.duration_emb(items[:, 2].unsqueeze(-1).float())

        x = torch.cat(
            (
                user_emb,
                gender_emb,
                age_emb,
                item_emb,
                item_pre_emb,
                source_emb,
                duration_emb,
            ),
            dim=-1,
        )
        return self.main(x).squeeze()

    def params_count(self) -> int:
        return sum([p.numel() for p in self.parameters() if p.requires_grad])


class LightningModel(pl.LightningModule):
    def __init__(
        self,
        user_size: int,
        item_size: int,
        source_size: int,
        pretrained_emb: torch.tensor,
        user_emb_size: int = 64,
        item_emb_size: int = 64,
        source_emb_size: int = 8,
        gender_emb_size: int = 8,
        duration_emb_size: int = 8,
        age_emb_size: int = 8,
        dropout: float = 0.4,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model = RecsysModel(
            user_size,
            item_size,
            source_size,
            pretrained_emb,
            user_emb_size,
            item_emb_size,
            source_emb_size,
            gender_emb_size,
            duration_emb_size,
            age_emb_size,
            dropout,
        )
        self.criterion = nn.MSELoss(reduction="none")
        self.save_hyperparameters()

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        users, items, target, weights = batch
        preds = self.model(users, items)
        train_loss = (weights * self.criterion(preds, target.float())).mean()
        self.log(
            "train_loss",
            train_loss,
            on_epoch=True,
            on_step=True,
            logger=True,
            prog_bar=True,
        )
        return train_loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        users, items, target, weights = batch
        preds = self.model(users, items)
        val_loss = (weights * self.criterion(preds, target.float())).mean()
        self.log(
            "val_loss",
            val_loss,
            on_epoch=True,
            on_step=False,
            logger=True,
            prog_bar=True,
        )

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> None:
        users, items = batch
        preds = self.model(users, items)
        with open("outs.txt", "w") as file:
            for i in range(len(preds)):
                file.write(f"{users[i][0].item()},{items[i][0].item()},{preds[i]}\n")

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            [
                {
                    "params": filter(
                        lambda p: p.requires_grad, self.model.parameters()
                    ),
                    "name": "model",
                    "weight_decay": 0.1,
                }
            ]
        )
        scheduler = LRCosineScheduler(
            optimizer=optimizer,
            warmup_steps=8002,
            cycle_steps=64016,
            cycle_mult=1,
            max_lr=0.002,
            min_lr=0.000002,
            gamma=1,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }


@hydra.main(version_base="1.3", config_path="./config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    wandb.login()
    items = pd.read_parquet(cfg.data.items_path)
    embeddings = torch.from_numpy(np.stack(items["embeddings"]))
    datamodule = DataModule(**cfg.data)
    model = LightningModel(pretrained_emb=embeddings, **cfg.model)
    trainer = pl.Trainer(
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath=".",
                filename="epoch_{epoch:02d}-{val_recall_100:.3f}-{val_loss:.3f}",
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                auto_insert_metric_name=True,
                save_weights_only=False,
            ),
            pl.callbacks.LearningRateMonitor(logging_interval="step"),
            pl.callbacks.TQDMProgressBar(
                leave=False,
                refresh_rate=20,
            ),
        ],
        logger=pl.loggers.wandb.WandbLogger(project="Vk RecSys"),
        **cfg.trainer,
    )
    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(model=model, datamodule=datamodule, ckpt_path="best")
    wandb.finish()


if __name__ == "__main__":
    main()
