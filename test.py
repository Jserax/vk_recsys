import hydra
import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
import torch.optim
from omegaconf import DictConfig

from train import DataModule, LightningModel


@hydra.main(version_base="1.3", config_path="./config", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
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
        **cfg.trainer,
    )
    datamodule.setup()
    datalodaer = datamodule.test_dataloader()
    with open("outs.txt", "a") as file:
        for users, items in datalodaer:
            preds = model.model(users, items)
            for i in range(len(preds)):
                file.write(f"{users[i][0].item()},{items[i][0].item()},{preds[i]}\n")


if __name__ == "__main__":
    main()
