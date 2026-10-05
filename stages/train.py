#!/usr/bin/env python
"""Train v2 weights from the feature tables, recorded in CMF.

Inputs: data/features/ece.parquet and data/features/pr.parquet (from
stages/features.py) and the label sheet they were joined with. Outputs: the
eight files the package's Ensemble loader expects, under models/v2/, plus
train/test accuracy per model and the ensemble accuracy on held-out shots.

The shot split is seeded and written to models/v2/split.json. One shared
split: the ECE and PR models are each trained on all of their own rows for
the training shots, and the ensemble is scored on held-out shots that have
both.

    pixi run train
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from hmode_classifier import Ensemble_Classifier
from hmode_classifier.cmf import ROOT, CmfStage
from hmode_classifier.features import ensemble_rows
from hmode_classifier.train import (N_CLUSTERS, accuracy, feature_columns, fit_classifier,
                                    fit_clusters, split_shots, to_onnx)

FEATURES = os.path.join(ROOT, "data", "features")
LABELS = os.path.join(ROOT, "data", "labels_sheet", "Labels.xlsx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--fraction", type=float, default=0.8)
    parser.add_argument("--out", default=os.path.join(ROOT, "models", "v2"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    stage = CmfStage("train", properties={"seed": args.seed, "train_fraction": args.fraction,
                                          "n_clusters": N_CLUSTERS, "weights_version": os.path.basename(args.out)})
    tables = {name: pd.read_parquet(os.path.join(FEATURES, f"{name}.parquet")) for name in ("ece", "pr")}
    for name in tables:
        stage.input(os.path.join(FEATURES, f"{name}.parquet"))
    stage.input(LABELS)

    shots = sorted(set(tables["ece"]["shot"]) | set(tables["pr"]["shot"]))
    train_shots, test_shots = split_shots(shots, args.fraction, args.seed)
    with open(os.path.join(args.out, "split.json"), "w") as f:
        json.dump({"seed": args.seed, "fraction": args.fraction, "train": train_shots, "test": test_shots}, f)

    metrics = {}
    fitted = {}
    for name, tag in (("ece", "ECE"), ("pr", "PR")):
        df = tables[name]
        train = df[df["shot"].isin(train_shots)]
        test = df[df["shot"].isin(test_shots)]
        clf = fit_classifier(train)
        km, centres, mean_dist = fit_clusters(train)
        width = len(feature_columns(df))
        with open(os.path.join(args.out, f"{tag}_GradBoost_v1.onnx"), "wb") as f:
            f.write(to_onnx(clf, width))
        with open(os.path.join(args.out, f"{tag}_Kmeans_v1.onnx"), "wb") as f:
            f.write(to_onnx(km, width))
        np.save(os.path.join(args.out, f"Cluster_loc_{tag}_v1.npy"), centres)
        np.save(os.path.join(args.out, f"Ave_Dist_{tag}_v1.npy"), np.float64(mean_dist))
        fitted[name] = clf
        metrics.update({f"{name}_train_rows": len(train), f"{name}_test_rows": len(test),
                        f"{name}_train_accuracy": accuracy(clf, train), f"{name}_test_accuracy": accuracy(clf, test)})
        print(f"{name}: train {len(train)} rows acc {metrics[f'{name}_train_accuracy']:.4f}, "
              f"test {len(test)} rows acc {metrics[f'{name}_test_accuracy']:.4f}", flush=True)

    # Ensemble accuracy on held-out shots that have both feature sets.
    both = ensemble_rows(tables["ece"], tables["pr"])
    both = both[both["shot"].isin(test_shots)]
    M = Ensemble_Classifier.load_models(args.out)
    xe = both[feature_columns(tables["ece"])].to_numpy(np.float32)
    xp = both[feature_columns(tables["pr"])].to_numpy(np.float32)
    pe = fitted["ece"].predict_proba(xe)[:, 1]
    pp = fitted["pr"].predict_proba(xp)[:, 1]
    ce = M["kmeans_ece"].run(None, {"input": xe})[0]
    cp = M["kmeans_pr"].run(None, {"input": xp})[0]
    de = np.linalg.norm(xe - M["cluster_ece"][ce], axis=1)
    dp = np.linalg.norm(xp - M["cluster_pr"][cp], axis=1)
    _, label = Ensemble_Classifier.Ensemble_Model(pe, pp, M["ave_ece"], M["ave_pr"], de, dp)
    metrics["ensemble_test_rows"] = len(both)
    metrics["ensemble_test_accuracy"] = float(np.mean(label == both["label"].to_numpy()))
    print(f"ensemble: test {len(both)} rows acc {metrics['ensemble_test_accuracy']:.4f}", flush=True)

    stage.output(args.out, weights_version=os.path.basename(args.out))
    stage.metrics("accuracy", metrics)
    stage.finalize()


if __name__ == "__main__":
    main()
