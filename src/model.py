#!/usr/bin/env python3.6
import timm
import torch
import torch.nn as nn
from torchvision import models

from src.backbones import BackboneType


class MRNet(nn.Module):

    LINEAR_CLASSIFIER_IN = 256
    KERNEL_SIZE = 7
    STRIDE = None
    PADDING = 0

    def _get_backbone(self):
        return models.alexnet(pretrained=True).features

    def __init__(self):
        super().__init__()

        self.backbone = self._get_backbone()
        self.fc = nn.Linear(self.LINEAR_CLASSIFIER_IN, 1)

        self.avg_pool = nn.AvgPool2d(
            kernel_size=self.KERNEL_SIZE, stride=self.STRIDE, padding=self.PADDING
        )
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, batch):
        batch_out = torch.tensor([]).to(batch.device)

        for series in batch:
            out = torch.tensor([]).to(batch.device)
            for image in series:
                out = torch.cat((out, self.backbone(image.unsqueeze(0))), 0)

            out = self.avg_pool(out).squeeze()
            out = out.max(dim=0, keepdim=True)[0].squeeze()

            out = self.fc(self.dropout(out))

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


class InceptionMRNetMixin(MRNet):
    KERNEL_SIZE = 5
    TIMM_MODEL_NAME = None

    def _get_backbone(self):
        model = timm.create_model(self.TIMM_MODEL_NAME, pretrained=True, num_classes=0, global_pool="")
        model.eval()
        return model


class MRNetOnResnet(InceptionMRNetMixin):
    LINEAR_CLASSIFIER_IN = 512
    KERNEL_SIZE = 7

    TIMM_MODEL_NAME = "resnet18"



class MRNetOnInceptionV4(InceptionMRNetMixin):
    LINEAR_CLASSIFIER_IN = 1536

    def _get_backbone(self):
        model = timm.create_model('inception_v4', pretrained=True, num_classes=0, global_pool="")
        model.eval()
        return model


class MRNetOnInceptionResnet(InceptionMRNetMixin):
    LINEAR_CLASSIFIER_IN = 1536

    def _get_backbone(self):
        model = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=0, global_pool="")
        model.eval()
        return model

class MRNetOnEfficientNet(InceptionMRNetMixin):
    LINEAR_CLASSIFIER_IN = 1280

    def _get_backbone(self):
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0, global_pool="")
        model.eval()
        return model


class MRNetOnFBNet(InceptionMRNetMixin):
    LINEAR_CLASSIFIER_IN = 1984

    def _get_backbone(self):
        model = timm.create_model('fbnetc_100', pretrained=True, num_classes=0, global_pool="")
        model.eval()
        return model



BACKBONE_MAPPING = {
    BackboneType.ALEXNET: MRNet,
    BackboneType.VGG_11: MRNetOnVGG11,
    BackboneType.VGG_16: MRNetOnVGG16,
    BackboneType.RESNET: MRNetOnResnet,
    BackboneType.INCEPTION_V4: MRNetOnInceptionV4,
    BackboneType.INCEPTION_RESNET: MRNetOnInceptionResnet,
    BackboneType.EFFICIENTNET: MRNetOnEfficientNet,
    BackboneType.FBNET: MRNetOnFBNet,
    # BackboneType.XCEPTION: MRNetOnXception,
}
