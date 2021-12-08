"""
The aim of this work is to train the MRNet architecture on different types of backbones
and compare them by accuracy, model size, training time, etc.

This module contains a list of backbones and additional features for them.
"""

from enum import Enum
from torchvision import transforms


class BackboneType(str, Enum):
    ALEXNET = "alexnet"
    VGG_11 = "vgg11"
    VGG_16 = "vgg16"
    INCEPTION = "inception"


ADDITIONAL_TRANSFORMS = {
    BackboneType.INCEPTION: {
        "pre": [
            transforms.Resize(299),
            transforms.CenterCrop(299),
        ]
    }
}
