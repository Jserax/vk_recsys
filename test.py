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
    model = LightningModel.load_from_checkpoint(
        "/kaggle/working/epoch_epoch=00-val_recall_100=0.000-val_loss=0.428.ckpt"
    )
    datamodule.setup()
    datalodaer = datamodule.test_dataloader()
    with open("outs.txt", "a") as file:
        for users, items in datalodaer:
            preds = model.model(users.cuda(), items.cuda())
            for i in range(len(preds)):
                file.write(f"{users[i][0].item()},{items[i][0].item()},{preds[i]}\n")


if __name__ == "__main__":
    main()
