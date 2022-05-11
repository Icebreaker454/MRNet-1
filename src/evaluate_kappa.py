""" Script for Cohens Kappa evaluation. Plots images per diagnosis """
import os


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn import metrics


VALIDATION_LABELS = "./valid_labels.csv"
VALIDATION_PATHS = "./valid_paths.csv"

MODELS = {
    "alexnet": "./alexnet_predictions.csv",
    "vgg11": "./vgg11_predictions.csv",
    "vgg16": "./vgg16_predictions.csv",
    "resnet": "./resnet_8e_predictions.csv",
    "efficientnet": "./efficientnet_predictions.csv",                
}

# TODO: Fix this ?
CONFIDENCE_RATES = [
    0.5, 0.6, 0.7, 0.8, 0.9
]


def report_model_kappa(predictions_df, valid_df, cases):
    """ For a given model predictions and validation set, 
    return the Cohen's kappa for each diagnosis, with EACH confidence rate
    """

    diagnoses = valid_df.columns.values[1:]

    ys = []
    Xs = []

    for i, case in enumerate(cases):
        case_row = valid_df[valid_df.case == int(case)]

        y = case_row.values[0, 1:].astype(np.float32)
        ys.append(y)

        X = predictions_df.iloc[i].values
        Xs.append(X)

    ys = np.asarray(ys).transpose()
    Xs = np.asarray(Xs).transpose()

    def deterministic_factory(rate):
        def deterministic(x):
            return 1 if x >= rate else 0
        return deterministic

    scores = {}

    for i, diagnosis in enumerate(diagnoses):

        for confidence in CONFIDENCE_RATES:
            transform = np.vectorize(deterministic_factory(confidence))

            discrete_X = transform(Xs[i].copy())

            scores.setdefault(diagnosis, []).append(
                metrics.cohen_kappa_score(
                    ys[i], discrete_X
                )
            )
    return scores


def main():
    print("Reporting Cohen's Kappa scores...")

    valid_df = pd.read_csv(VALIDATION_LABELS)    

    old_case = None
    cases = []
    with open(VALIDATION_PATHS, "r") as paths:
        for path in paths:
            case = os.path.splitext(os.path.basename(path.strip()))[0]
            if case == old_case:
                next
            else:
                cases.append(case)
                old_case = case

    scores = {}
    for name, predictions_path in MODELS.items():

        print(f"Reporting model {name} Kappa scores")

        pred_df = pd.read_csv(predictions_path, header=None)

        
        scores[name] = report_model_kappa(pred_df, valid_df, cases)

        print(f"Got scores: {scores}")


    diagnoses = ["abnormal", "acl", "meniscus"]

    for diagnosis in diagnoses:

        for model in MODELS:
            plt.plot(CONFIDENCE_RATES, scores[model][diagnosis], label=model)

        plt.ylabel("Kappa score")
        plt.xlabel("Confidence")
        
        plt.title(f"Combined Kappa score for {diagnosis} diagnosis")
        plt.legend()

        plt.savefig(f"./results_kappa_{diagnosis}.png", dpi=300)
        plt.clf()


if __name__ == "__main__":
    main()
