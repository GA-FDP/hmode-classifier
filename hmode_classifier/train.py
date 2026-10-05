"""Fit and export the classifiers.

Hyperparameters are the ones in the original training notebook: a histogram
gradient-boosted classifier (learning rate 0.1, 100 iterations, 31 leaves, no
L2) per diagnostic, and a 200-cluster k-means per diagnostic whose cluster
centres and mean in-cluster distance weight the ensemble.
"""
import random

import numpy as np
from skl2onnx import to_onnx as _to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier

GRADBOOST = dict(learning_rate=0.1, max_iter=100, max_leaf_nodes=31, l2_regularization=0.0)
N_CLUSTERS = 200
ONNX_OPSET = 17


def feature_columns(df):
    return [c for c in df.columns if c not in ("shot", "time", "label")]


def split_shots(shots, fraction=0.8, seed=0):
    """Train/test shot lists; the same split for the same seed."""
    shots = sorted(int(s) for s in shots)
    rng = random.Random(seed)
    train = sorted(rng.sample(shots, int(fraction * len(shots))))
    test = [s for s in shots if s not in set(train)]
    return train, test


def fit_classifier(df):
    x = df[feature_columns(df)].to_numpy(np.float32)
    return HistGradientBoostingClassifier(**GRADBOOST).fit(x, df["label"].to_numpy())


def fit_clusters(df, n_clusters=N_CLUSTERS, seed=0):
    """k-means on the training rows, with the statistics the ensemble needs."""
    x = df[feature_columns(df)].to_numpy(np.float32)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit(x)
    centres = km.cluster_centers_.astype(np.float64)
    distances = np.linalg.norm(x - centres[km.labels_], axis=1)
    return km, centres, float(np.mean(distances))


def to_onnx(model, width):
    """ONNX bytes with the same input/output names and shapes as the v1 files."""
    onx = _to_onnx(model, initial_types=[("input", FloatTensorType([None, width]))], target_opset=ONNX_OPSET)
    return onx.SerializeToString()


def accuracy(clf, df):
    x = df[feature_columns(df)].to_numpy(np.float32)
    return float(np.mean(clf.predict(x) == df["label"].to_numpy()))
