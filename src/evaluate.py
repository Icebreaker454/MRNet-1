#!/usr/bin/env python3.6
"""Calculates the average AUC score of the abnormality detection, ACL tear and
Meniscal tear tasks.

Usage:
  evaluate.py <valid_paths_csv> <preds_csv> <valid_labels_csv> [options]
  evaluate.py (-h | --help)

General options:
  -h --help          Show this screen.

Arguments:
  <valid_paths_csv>    csv file listing paths to validation set, which needs to
                       be in a specific order - an example is provided as
                       valid-paths.csv in the root of the project
                       e.g. 'valid-paths.csv'
  <preds_csv>          csv file generated by src/predict.py
                       e.g. 'out_dir/predictions.csv'
  <valid_labels_csv>   csv file containing labels for the valid dataset
                       e.g. 'MRNet-v1.0/valid_labels.csv'

Training options:
  --backbone=<backbone> Backbone used. "alexnet", "vgg16" or "inception"
  --plot-auc=<plot_auc> Plot AUC or not

"""

import os
import sys
import csv
from docopt import docopt
from pprint import pprint

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics

from backbones import BackboneType


def main(valid_paths_csv, preds_csv, valid_labels_csv, backbone: BackboneType = None, plot_auc: bool = False):
    print(f"Reporting {preds_csv} scores...")

    preds_df = pd.read_csv(preds_csv, header=None)
    valid_df = pd.read_csv(valid_labels_csv)

    old_case = None

    cases = []
    with open(valid_paths_csv, "r") as paths:
        for path in paths:
            case = os.path.splitext(os.path.basename(path.strip()))[0]
            if case == old_case:
                next
            else:
                cases.append(case)
                old_case = case

    ys = []
    Xs = []

    for i, case in enumerate(cases):
        case_row = valid_df[valid_df.case == int(case)]

        y = case_row.values[0, 1:].astype(np.float32)
        ys.append(y)

        X = preds_df.iloc[i].values
        Xs.append(X)

    ys = np.asarray(ys).transpose()
    Xs = np.asarray(Xs).transpose()

    aucs = {}

    THRESHOLDS = {
        "50%": 0.5,
        # "80%": 0.8,
        # "95%": 0.95,
    }
        

    accuracy = {}

    f1 = {}    
    
    diagnoses = valid_df.columns.values[1:]

    for i, diagnosis in enumerate(diagnoses):
        auc = metrics.roc_auc_score(ys[i], Xs[i])

        for name, value in THRESHOLDS.items():
            X_threshold = np.vectorize(lambda x: 1 if x >= value else 0)(Xs[i])
            acc = metrics.accuracy_score(ys[i], X_threshold)

            f1_score = metrics.f1_score(ys[i], X_threshold)            

            acc_dict = accuracy.setdefault(name, {})
            acc_dict[diagnosis] = acc

            f1_dict = f1.setdefault(name, {})
            f1_dict[diagnosis] = f1_score

        if plot_auc:
            name = f"{backbone}-{diagnosis}"
            metrics.RocCurveDisplay.from_predictions(
                ys[i], Xs[i], name=name,
            )
            plt.savefig(f"./auc_images/{name}.png")

        aucs[diagnosis] = auc

    aucs["avegare"] = np.array(list(aucs.values())).mean()

    print("AUCs:")
    for k, v in aucs.items():
        print(f"  {k}: {v:.3f}")

    print("========================================")        

    for name, value in THRESHOLDS.items():
        print(f"Accuracy threshold={value}")

        for diag, v in accuracy[name].items():
            print(f"  {diag}: {v:.3f}")
        
        print(f"Average: {np.array(list(accuracy[name].values())).mean()}")

        print("========================================")
        print(f"F1 Score: threshold={value}")
        for diag, v in f1[name].items():
            print(f"  {diag}: {v:.3f}")
                
        print(f"Average: {np.array(list(f1[name].values())).mean()}")
        
        

if __name__ == "__main__":
    arguments = docopt(__doc__)

    main(
        arguments["<valid_paths_csv>"],
        arguments["<preds_csv>"],
        arguments["<valid_labels_csv>"],
        arguments["--backbone"],
        arguments["--plot-auc"],        
    )
