import os
from glob import glob
from typing import Any, List, Union

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

from src.utils import preprocess_data
from src.backbones import ADDITIONAL_TRANSFORMS, BackboneType


class MRNetDataset(Dataset):
    def __init__(
        self,
        dataset_dir: str,
        labels_path: str,
        plane: str,
        transform: List[Any] = None,
        device: Union[torch.device, str] = None,
    ):
        self.case_paths = sorted(glob(f"{dataset_dir}/{plane}/**.npy"))
        self.labels_df = pd.read_csv(labels_path)
        self.transform = transform
        self.window = 7
        self.device = device
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def __len__(self):
        return len(self.case_paths)

    def __getitem__(self, idx):
        case_path = self.case_paths[idx]
        series = preprocess_data(case_path, self.transform)

        case_id = int(os.path.splitext(os.path.basename(case_path))[0])
        case_row = self.labels_df[self.labels_df.case == case_id]
        diagnoses = case_row.values[0, 1:].astype(np.float32)
        labels = torch.tensor(diagnoses)

        return (series, labels)


def make_dataset(
    data_dir: str,
    dataset_type: str,
    plane: str,
    device: Union[torch.device, str] = None,
    backbone: BackboneType = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_dir = f"{data_dir}/{dataset_type}"
    labels_path = f"{data_dir}/{dataset_type}_labels.csv"

    if dataset_type == "train":

        steps = [
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(25, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ]
        if backbone is not None and ADDITIONAL_TRANSFORMS.get(backbone):
            additional = ADDITIONAL_TRANSFORMS[backbone]
            steps = [
                *additional.get("pre", []),
                *steps,
                *additional.get("post", []),
            ]

        transform = transforms.Compose(steps)

    elif dataset_type == "valid":
        transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])
    else:
        raise ValueError("Dataset needs to be train or valid.")

    dataset = MRNetDataset(
        dataset_dir, labels_path, plane, transform=transform, device=device
    )

    return dataset
