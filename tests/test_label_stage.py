import numpy as np

from stages.label import labels_frame


def test_labels_frame_flattens_the_three_results():
    ece = [{"shot": 1, "Label_Times": np.array([100, 200]), "LH_Label": np.array([0, 1]),
            "LH_Prob": [{0: 0.9, 1: 0.1}, {0: 0.2, 1: 0.8}]}]
    pr = [{"shot": 1, "Label_Times": np.array([200]), "LH_Label": np.array([1]), "LH_Prob": [{0: 0.3, 1: 0.7}]}]
    ens = {1: {"shot": 1, "Time": np.array([200]), "Ensemble_Hmode_Prob": np.array([0.75]),
               "Ensemble_Label": np.array([1.0]), "Error": ""}}
    df = labels_frame(ece, pr, ens)
    assert df.columns.tolist() == ["model", "shot", "time", "label", "h_prob"]
    assert df[df.model == "ece"]["h_prob"].tolist() == [0.1, 0.8]
    assert df[df.model == "ensemble"]["label"].tolist() == [1]
