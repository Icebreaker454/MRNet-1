import os
import csv
import random
from pathlib import Path

import glob

from google.cloud import storage
import numpy as np
import torch
from sklearn import metrics

MAX_PIXEL_VAL = 255
MEAN = 58.09
STD = 49.73


def seed_all(seed: int):
    """ Seed for each and every possible source of randomness """

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_mrnet_dataset(data_dir):
    """Load MRNet dataset inside a DATA dir"""

    prefix = "MRNet-v1.0"

    path = Path(data_dir)
    path.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket("mrnet-training-bucket")

    blobs = [blob for blob in bucket.list_blobs(prefix=prefix)]
    paths = {}

    with client.batch():
        for blob in blobs:
            path, filename = blob.name.rsplit("/", 1)
            if path.startswith(prefix):
                path = path[len(prefix):]

            path = path.lstrip("/")

            local_path = Path(data_dir, path)
            if local_path not in paths:
                local_path.mkdir(parents=True, exist_ok=True)

            print(f"Downloading blob {blob} to path {local_path / filename}")
            blob.download_to_filename(local_path / filename)


def save_model_to_gs(out_dir: str, epochs: int, plane: str, backbone: str):
    experiment_dir = f"{backbone}_cnn_{epochs}e"

    local_files = glob.glob(f"{out_dir}/{backbone}_cnn_{plane}*.pt")

    bucket = storage.Client().bucket("mrnet-training-bucket")

    for file in local_files:
        filename = file.rsplit("/", 1)[1]
        blob = bucket.blob(f"{experiment_dir}/{plane}/{filename}")
        blob.upload_from_filename(file)


def preprocess_data(case_path, transform=None):
    series = np.load(case_path).astype(np.float32)
    series = torch.tensor(np.stack((series,) * 3, axis=1))

    if transform is not None:
        for i, slice in enumerate(series.split(1)):
            series[i] = transform(slice.squeeze())

    series = (series - series.min()) / (series.max() - series.min()) * MAX_PIXEL_VAL
    series = (series - MEAN) / STD

    return series


def create_output_dir(exp, plane):
    out_dir = f"/tmp/models/{exp}"
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    losses_path = create_losses_csv(out_dir, plane)

    return out_dir, losses_path


def create_losses_csv(out_dir, plane):
    losses_path = f"{out_dir}/losses_{plane}.csv"

    with open(f"{losses_path}", mode="w") as losses_csv:
        fields = [
            "t_abnormal",
            "t_acl",
            "t_meniscus",
            "v_abnormal",
            "v_acl",
            "v_meniscus",
        ]
        writer = csv.DictWriter(losses_csv, fieldnames=fields)
        writer.writeheader()

    return losses_path


def calculate_aucs(all_labels, all_preds):
    all_labels = np.array(all_labels).transpose()
    all_preds = np.array(all_preds).transpose()

    aucs = [
        metrics.roc_auc_score(labels, preds)
        for labels, preds in zip(all_labels, all_preds)
    ]

    return aucs


def print_stats(batch_train_losses, batch_valid_losses, valid_labels, valid_preds):
    aucs = calculate_aucs(valid_labels, valid_preds)

    print(
        f"Train losses - abnormal: {batch_train_losses[0]:.3f},",
        f"acl: {batch_train_losses[1]:.3f},",
        f"meniscus: {batch_train_losses[2]:.3f}",
        f"\nValid losses - abnormal: {batch_valid_losses[0]:.3f},",
        f"acl: {batch_valid_losses[1]:.3f},",
        f"meniscus: {batch_valid_losses[2]:.3f}",
        f"\nValid AUCs - abnormal: {aucs[0]:.3f},",
        f"acl: {aucs[1]:.3f},",
        f"meniscus: {aucs[2]:.3f}",
    )


def save_losses(train_losses, valid_losses, losses_path):
    with open(f"{losses_path}", mode="a") as losses_csv:
        writer = csv.writer(losses_csv)
        writer.writerow(np.append(train_losses, valid_losses))


def save_checkpoint(epoch, plane, diagnosis, model, backbone, optimizer, out_dir):
    print(f"Min valid loss for {diagnosis}, saving the checkpoint...")

    checkpoint = {
        "epoch": epoch,
        "plane": plane,
        "diagnosis": diagnosis,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }

    chkpt = f"{backbone}_cnn_{plane}_{diagnosis}_{epoch:02d}.pt"
    torch.save(checkpoint, f"{out_dir}/{chkpt}")
