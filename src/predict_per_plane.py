#!/usr/bin/env python3.6
"""Calculates predictions on the validation dataset, using CNN models specified
in src/cnn_models_paths.txt

Usage:
  predict_per_plane.py <valid_paths_csv> <cnn_models_paths> <output_dir> [options]
  predict_per_plane.py (-h | --help)

General options:
  -h --help          Show this screen.

Arguments:
  <valid_paths_csv>  csv file listing paths to validation set, which needs to
                     be in a specific order - an example is provided as
                     valid-paths.csv in the root of the project
                     e.g. 'valid-paths.csv'
  <cnn_models_paths> .txt file listing the cnn models which should be loaded
                     for predictions
  <output_dir>       Directory where predictions are saved as a 3-column csv
                     file (with no header), where each column contains a
                     prediction for abnormality, ACL tear, and meniscal tear,
                     in that order
                     e.g. 'out_dir
Training options:
  --backbone=<backbone> Backbone used. "alexnet", "vgg16" or "inception"

"""

import os
import csv
from tqdm import tqdm
from docopt import docopt

import torch
import numpy as np
import pandas as pd
import joblib
from torchvision import transforms

from src.backbones import BackboneType
from src.model import MRNet, BACKBONE_MAPPING
from src.utils import preprocess_data


def main(
    valid_paths_csv: str,
    cnn_models_paths: str,
    output_dir: str,
    backbone: BackboneType = None,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_files_df = pd.read_csv(valid_paths_csv, header=None)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load MRNet models
    print(f"Loading CNN models listed in {cnn_models_paths}...")

    cnn_models_paths = [line.rstrip("\n") for line in open(cnn_models_paths, "r")]

    abnormal_mrnets = []
    acl_mrnets = []
    meniscus_mrnets = []

    model_class = BACKBONE_MAPPING.get(backbone, MRNet)

    for i, mrnet_path in enumerate(cnn_models_paths):
        model = model_class().to(device)
        checkpoint = torch.load(mrnet_path, map_location=device)

        if backbone == "alexnet":
            keys = map(lambda k: k.replace("alexnet", "backbone") if "alexnet" in k else k, checkpoint["state_dict"])

            checkpoint["state_dict"] = dict(
                zip(keys, checkpoint["state_dict"].values())
            )

        model.load_state_dict(checkpoint["state_dict"])

        if i < 3:
            abnormal_mrnets.append(model)
        elif i >= 3 and i < 6:
            acl_mrnets.append(model)
        else:
            meniscus_mrnets.append(model)

    mrnets = [abnormal_mrnets, acl_mrnets, meniscus_mrnets]

    # Parse input, 3 rows at a time (i.e. per case)

    npy_paths = [row.values[0] for _, row in input_files_df.iterrows()]

    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor()])

    output_file_template = f"./predictions_{backbone}_{{plane}}.csv"

    print(f"Generating predictions per case for each plane...")
    print(f"Predictions will be saved as {output_file_template}")


    preds = {
        "sagittal": [],
        "coronal": [],
        "axial": [],
    }

    for i in tqdm(range(0, len(npy_paths), 3)):
        case_paths = [npy_paths[i], npy_paths[i + 1], npy_paths[i + 2]]

        data = []

        for case_path in case_paths:
            series = preprocess_data(case_path, transform)
            data.append(series.unsqueeze(0).to(device))

        # Make predictions per case

        axial_pred = []
        coronal_pred = []
        sagittal_pred = []

        for i, mrnet in enumerate(mrnets):  # For each condition (mrnet)
            # Based on each plane (data)
            sagittal_pred.append(
                torch.sigmoid(mrnet[0](data[0]).detach().cpu()).item()
            )
            coronal_pred.append(
                torch.sigmoid(mrnet[1](data[1]).detach().cpu()).item()
            )
            axial_pred.append(
                torch.sigmoid(mrnet[2](data[2]).detach().cpu()).item()
            )

        preds["sagittal"].append(sagittal_pred)
        preds["coronal"].append(coronal_pred)
        preds["axial"].append(axial_pred)

    for plane in ("axial", "coronal", "sagittal"):
        with open(output_file_template.format(plane=plane), "w") as f:
            writer = csv.writer(f)
            for pred in preds[plane]:
                writer.writerow(pred)




if __name__ == "__main__":
    arguments = docopt(__doc__)

    print("Parsing arguments...")

    main(
        arguments["<valid_paths_csv>"],
        arguments["<cnn_models_paths>"],
        arguments["<output_dir>"],
        arguments["--backbone"],
    )
