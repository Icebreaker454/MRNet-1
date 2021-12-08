#!/usr/bin/env python3.6

import torch
import torch.nn as nn
from torchvision import models

from backbones import BackboneType


class MRNet(nn.Module):

    LINEAR_CLASSIFIER_IN = 256

    def _get_backbone(self):
        return models.alexnet(pretrained=True).features

    def __init__(self):
        super().__init__()

        self.backbone = self._get_backbone()
        self.fc = nn.Linear(self.LINEAR_CLASSIFIER_IN, 1)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=None, padding=0)
        self.dropout = nn.Dropout(p=0.5)

    @property
    def features(self):
        return self.backbone

    @property
    def classifier(self):
        return self.fc

    def forward(self, batch):
        batch_out = torch.tensor([]).to(batch.device)

        for series in batch:
            out = torch.tensor([]).to(batch.device)
            for image in series:
                out = torch.cat((out, self.features(image.unsqueeze(0))), 0)

            out = self.avg_pool(out).squeeze()
            out = out.max(dim=0, keepdim=True)[0].squeeze()

            out = self.classifier(self.dropout(out))

            batch_out = torch.cat((batch_out, out), 0)

        return batch_out


class MRNetOnVGG16(MRNet):

    LINEAR_CLASSIFIER_IN = 512

    def _get_backbone(self):
        return models.vgg16(pretrained=True).features


class MRNetOnVGG11(MRNet):

    LINEAR_CLASSIFIER_IN = 512

    def _get_backbone(self):
        return models.vgg11(pretrained=True).features


class MRNetOnInceptionV3(MRNet):
    # NOTE: The minimal working image dimensions for Inception are 299x299
    LINEAR_CLASSIFIER_IN = 299

    def _get_backbone(self):
        return models.inception_v3(pretrained=True)

    @property
    def features(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

        layers = list(self.backbone.children())
        return torch.nn.Sequential(*layers[:-2])


BACKBONE_MAPPING = {
    BackboneType.ALEXNET: MRNet,
    BackboneType.VGG_11: MRNetOnVGG11,
    BackboneType.VGG_16: MRNetOnVGG16,
    BackboneType.INCEPTION: MRNetOnInceptionV3,
}
