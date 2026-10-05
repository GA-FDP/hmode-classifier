import numpy as np
import onnxruntime as ort

from hmode_classifier import MODEL_DIR, Hmode_Ml_Predictions

EXPECTED = {
    "ECE_GradBoost_v1.onnx", "ECE_Kmeans_v1.onnx", "PR_GradBoost_v1.onnx", "PR_Kmeans_v1.onnx",
    "Ave_Dist_ECE_v1.npy", "Ave_Dist_PR_v1.npy", "Cluster_loc_ECE_v1.npy", "Cluster_loc_PR_v1.npy",
}


def test_v1_weights_ship_with_the_package():
    assert {p.name for p in MODEL_DIR.iterdir()} >= EXPECTED


def test_v1_gradboost_models_score_a_feature_row():
    for name, width in (("ECE_GradBoost_v1.onnx", 7), ("PR_GradBoost_v1.onnx", 10)):
        session = ort.InferenceSession(str(MODEL_DIR / name))
        label, prob = session.run(None, {"input": np.zeros((1, width), np.float32)})
        assert label.shape == (1,)
        assert set(prob[0].keys()) == {0, 1}


def test_models_dir_override():
    hm = Hmode_Ml_Predictions("ECE", [1], [100], models_dir="/nonexistent/v2")
    assert str(hm.MODEL_DIR).endswith("v2")
