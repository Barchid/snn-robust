from typing import Optional
import pytorch_lightning as pl
from tonic.dataset import Dataset
from tonic.download_utils import download_url, extract_archive
from torch.functional import split
from torch.utils import data
from torch.utils.data import random_split, DataLoader
import tonic
from torchvision import transforms
import os
import numpy as np
from torchvision.datasets import CIFAR10


class CIFAR10_C_DataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, data_dir: str = "data", noise="brightness", **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir

        # create the directory if not exist
        os.makedirs(data_dir, exist_ok=True)

    def prepare_data(self) -> None:
        # download train set
        CIFAR10(self.data_dir, train=True, download=True)

        # download & extract all -C data as val set

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_set = tonic.datasets.NMNIST(save_to=self.data_dir, train=True, transform=self.transform)
        self.val_set = tonic.datasets.NMNIST(save_to=self.data_dir, train=False, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False)


class CIFAR10_C_Dataset(Dataset):
    """CIFAR-10 C(orruption) dataset"""

    DOWNLOAD_URL = "https://zenodo.org/record/2535967/files/CIFAR-10-C.tar?download=1"

    def __init__(self, data_dir="data/", noise="brightness", severity: int = 1):
        super(CIFAR10_C_Dataset, self).__init__()

        self._install_dataset(data_dir)

        dataset_dir = os.path.join(data_dir, 'CIFAR-10-C')

        # load data
        self.data = np.load(os.path.join(dataset_dir, f"{noise}.npy"))
        self.labels = np.load(os.path.join(dataset_dir, "labels.npy"))

        self.severity = severity

        self.transform = transforms.ToTensor()

    def _install_dataset(self, data_dir: str):
        """Downloads and extracts the CIFAR10-C dataset if needed.

        Args:
            data_dir (str): root data directory where the dataset is located
        """

        dataset_dir = os.path.join(data_dir, 'CIFAR-10-C')
        if os.path.exists(dataset_dir):
            print(f'Dataset already exists at location "{dataset_dir}". Skip download & install')
            return

        download_url(CIFAR10_C_Dataset.DOWNLOAD_URL, data_dir, filename="CIFAR-10-C.tar")

        # TODO: put to true when it's correctly tested
        extract_archive(os.path.join(data_dir, 'CIFAR-10-C.tar'), remove_finished=True)

    def __getitem__(self, index):
        real_index = (self.severity - 1) * 10000 + index

        image = self.data[real_index]
        image = self.transform(image)

        label = self.labels[real_index]

        return image, label

    def __len__(self):
        return 10000
