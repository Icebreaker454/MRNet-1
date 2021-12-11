#!/usr/bin/env python3.6
"""Trains three CNN models to predict abnormalities, ACL tears and meniscal
tears for a given plane (axial, coronal or sagittal) of knee MRI images.

Usage:
  train_cnn_models.py <data_dir> <plane> <epochs> [options]
  train_cnn_models.py (-h | --help)

General options:
  -h --help             Show this screen.

Arguments:
  <data_dir>            Path to a directory where the data lives e.g. 'MRNet-v1.0'
  <plane>               MRI plane of choice ('axial', 'coronal', 'sagittal')
  <epochs>              Number of epochs e.g. 50

Training options:
  --lr=<lr>             Learning rate for nn.optim.Adam optimizer [default: 0.00001]
  --backbone=<backbone> Backbone used. "alexnet", "vgg16" or "inception"
  --no-load-dataset     Disables downloading the dataset from Google cloud
  --weight-decay=<wd>   Weight decay for nn.optim.Adam optimizer [default: 0.01]
  --device=<device>     Device to run code ('cpu' or 'cuda') - if not provided,
                        it will be set to the value returned by torch.cuda.is_available()
"""

import sys
import numpy as np
import pandas as pd
from datetime import datetime

import click
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.backbones import BackboneType
from src.data_loader import make_data_loader
from src.model import MRNet, BACKBONE_MAPPING
from src.utils import (
    seed_all,
    create_output_dir,
    print_stats,
    save_losses,
    save_checkpoint,
    load_mrnet_dataset,
)


def calculate_weights(data_dir, dataset_type, device):
    diagnoses = ["abnormal", "acl", "meniscus"]

    labels_path = f"{data_dir}/{dataset_type}_labels.csv"
    labels_df = pd.read_csv(labels_path)

    weights = []

    for diagnosis in diagnoses:
        neg_count, pos_count = labels_df[diagnosis].value_counts().sort_index()
        weight = torch.tensor([neg_count / pos_count])
        weight = weight.to(device)
        weights.append(weight)

    return weights


def make_adam_optimizer(model, lr, weight_decay):
    return optim.Adam(model.parameters(), lr, weight_decay=weight_decay)


def make_lr_scheduler(optimizer, mode="min", factor=0.3, patience=1, verbose=False):
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=mode, factor=factor, patience=patience, verbose=verbose
    )


def batch_forward_backprop(models, inputs, labels, criterions, optimizers):
    losses = []

    for i, (model, label, criterion, optimizer) in enumerate(
        zip(models, labels[0], criterions, optimizers)
    ):
        model.train()
        optimizer.zero_grad()

        out = model(inputs)
        loss = criterion(out, label.unsqueeze(0))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return np.array(losses)


def batch_forward(models, inputs, labels, criterions):
    preds = []
    losses = []

    for i, (model, label, criterion) in enumerate(zip(models, labels[0], criterions)):
        model.eval()

        out = model(inputs)
        preds.append(out.item())
        loss = criterion(out, label.unsqueeze(0))
        losses.append(loss.item())

    return np.array(preds), np.array(losses)


def update_lr_schedulers(lr_schedulers, batch_valid_losses):
    for scheduler, v_loss in zip(lr_schedulers, batch_valid_losses):
        scheduler.step(v_loss)


@click.command()
@click.argument(
    "data_dir",
    type=str,
)
@click.argument("plane", type=click.Choice(["axial", "coronal", "sagittal"]))
@click.option("--epochs", type=int, default=10, help="Number of epochs to train")
@click.option("--lr", type=float, default=1e-5, help="Learning rate")
@click.option("--weight-decay", type=float, default=0.01, help="Weight decay")
@click.option("--backbone", type=BackboneType, help="Backbone net used for training")
@click.option("--seed", type=int, default=42, help="Seed used for training reproducibility")
@click.option(
    "--device",
    type=click.Choice(["cuda", "cpu"]),
    default="cuda",
    help="Training device",
)
@click.option("--load-dataset/--no-load-dataset", default=False)
def main(
    data_dir: str,
    plane: str,
    epochs: int,
    lr: float,
    weight_decay: float,
    backbone: BackboneType = None,
        seed: int = 42,
    device: str = None,
    load_dataset: bool = False,
):

    seed_all(seed)

    if load_dataset:
        load_mrnet_dataset(data_dir)

    diagnoses = ["abnormal", "acl", "meniscus"]

    exp = f"{backbone}-{epochs}e-{datetime.now():%Y-%m-%d_%H-%M}"
    out_dir, losses_path = create_output_dir(exp, plane)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device is: {device}")

    print("Creating data loaders...")

    train_loader = make_data_loader(
        data_dir, "train", plane, device, shuffle=True, backbone=backbone
    )
    valid_loader = make_data_loader(data_dir, "valid", plane, device, backbone=backbone)

    print(f"Creating models...")

    # Create a model for each diagnosis

    if backbone is None:
        backbone = "alexnet"

    model_class = BACKBONE_MAPPING.get(backbone, MRNet)

    models = [
        model_class().to(device),
        model_class().to(device),
        model_class().to(device),
    ]

    # Calculate loss weights based on the prevalences in train set

    pos_weights = calculate_weights(data_dir, "train", device)
    criterions = [nn.BCEWithLogitsLoss(pos_weight=weight) for weight in pos_weights]

    optimizers = [make_adam_optimizer(model, lr, weight_decay) for model in models]

    lr_schedulers = [make_lr_scheduler(optimizer) for optimizer in optimizers]

    min_valid_losses = [np.inf, np.inf, np.inf]

    print(f"Training a model using {plane} series...")
    print(f"Checkpoints and losses will be save to {out_dir}")

    for epoch, _ in enumerate(range(epochs), 1):
        print(f"=== Epoch {epoch}/{epochs} ===")

        batch_train_losses = np.array([0.0, 0.0, 0.0])
        batch_valid_losses = np.array([0.0, 0.0, 0.0])

        for inputs, labels in tqdm(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            batch_loss = batch_forward_backprop(
                models, inputs, labels, criterions, optimizers
            )
            batch_train_losses += batch_loss

        valid_preds = []
        valid_labels = []

        print(f"====Validation on epoch {epoch}=======")

        for inputs, labels in tqdm(valid_loader):
            inputs, labels = inputs.to(device), labels.to(device)

            batch_preds, batch_loss = batch_forward(models, inputs, labels, criterions)
            batch_valid_losses += batch_loss

            valid_labels.append(labels.detach().cpu().numpy().squeeze())
            valid_preds.append(batch_preds)

        batch_train_losses /= len(train_loader)
        batch_valid_losses /= len(valid_loader)

        print_stats(batch_train_losses, batch_valid_losses, valid_labels, valid_preds)
        save_losses(batch_train_losses, batch_valid_losses, losses_path)

        update_lr_schedulers(lr_schedulers, batch_valid_losses)

        for i, (batch_v_loss, min_v_loss) in enumerate(
            zip(batch_valid_losses, min_valid_losses)
        ):

            if batch_v_loss < min_v_loss:
                save_checkpoint(
                    epoch,
                    plane,
                    diagnoses[i],
                    models[i],
                    backbone,
                    optimizers[i],
                    out_dir,
                )

                min_valid_losses[i] = batch_v_loss

    save_models_to_gs(out_dir, epochs, plane, backbone)


if __name__ == "__main__":
    main()
