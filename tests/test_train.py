import numpy as np
import onnxruntime as ort
import pandas as pd

from hmode_classifier.train import fit_classifier, fit_clusters, split_shots, to_onnx


def _rows(n_shots=30, per_shot=10, width=7, seed=0):
    rng = np.random.default_rng(seed)
    shots = np.repeat(np.arange(n_shots), per_shot)
    x = rng.normal(size=(len(shots), width))
    y = (x[:, 0] > 0).astype(int)
    df = pd.DataFrame(x, columns=[f"e{i}" for i in range(width)])
    df.insert(0, "label", y)
    df.insert(0, "time", np.tile(np.arange(per_shot) * 100, n_shots))
    df.insert(0, "shot", shots)
    return df


def test_split_is_by_shot_and_deterministic():
    df = _rows()
    train_a, test_a = split_shots(df["shot"].unique(), fraction=0.8, seed=1)
    train_b, test_b = split_shots(df["shot"].unique(), fraction=0.8, seed=1)
    assert train_a == train_b and test_a == test_b
    assert len(train_a) == 24 and len(test_a) == 6
    assert not set(train_a) & set(test_a)


def test_onnx_matches_sklearn_probabilities():
    df = _rows()
    clf = fit_classifier(df)
    x = df[[c for c in df.columns if c.startswith("e")]].to_numpy(np.float32)
    session = ort.InferenceSession(to_onnx(clf, x.shape[1]))
    label, prob = session.run(None, {"input": x})
    assert np.array_equal(label, clf.predict(x))
    assert np.allclose([p[1] for p in prob], clf.predict_proba(x)[:, 1], atol=1e-5)


def test_clusters_report_centres_and_mean_distance():
    df = _rows()
    km, centres, mean_dist = fit_clusters(df, n_clusters=5)
    x = df[[c for c in df.columns if c.startswith("e")]].to_numpy(np.float32)
    session = ort.InferenceSession(to_onnx(km, x.shape[1]))
    cluster, _ = session.run(None, {"input": x})
    assert centres.shape == (5, 7)
    assert np.array_equal(cluster, km.predict(x))
    assert mean_dist > 0
