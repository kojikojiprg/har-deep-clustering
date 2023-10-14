from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader

from .collective_activity import CollectiveActivityDataset


class Datamodule(LightningDataModule):
    def __init__(
        self,
        dataset_dir: str,
        batch_size: int,
        seq_len: int,
        resize_ratio: float,
        stage: str,
        dataset_type: str,
    ):
        super().__init__()
        self._batch_size = batch_size

        if dataset_type == "collective":
            self._dataset = CollectiveActivityDataset(
                dataset_dir, seq_len, resize_ratio, stage
            )
        elif dataset_type == "volleyball":
            pass
        elif dataset_type == "video":
            pass
        else:
            raise ValueError

    @property
    def n_samples(self):
        return self._dataset.n_samples

    @property
    def n_samples_batch(self):
        return self._dataset.n_samples_batch

    def train_dataloader(self):
        return DataLoader(self._dataset, self._batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self):
        return DataLoader(self._dataset, self._batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self):
        return DataLoader(self._dataset, self._batch_size, shuffle=False, num_workers=8)

    def predict_dataloader(self):
        return DataLoader(self._dataset, self._batch_size, shuffle=False, num_workers=8)
