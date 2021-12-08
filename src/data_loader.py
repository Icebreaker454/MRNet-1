import torch
from typing import Union
from torch.utils.data import DataLoader

from src.dataset import make_dataset
from src.backbones import BackboneType


def make_data_loader(
    data_dir: str,
    dataset_type: str,
    plane: str,
    device: Union[torch.device, str] = None,
    shuffle: bool = False,
    backbone: BackboneType = None,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = make_dataset(
        data_dir, dataset_type, plane, device=device, backbone=backbone
    )
    data_loader = DataLoader(dataset, batch_size=1, shuffle=shuffle)

    return data_loader
