import numpy as np
import pandas as pd

from hmode_classifier.features import ensemble_rows, feature_table


class FakeRecord(dict):
    def __init__(self, shot, times, vectors):
        super().__init__(shot=shot, Label_Times=np.asarray(times), Feature_Vector=np.asarray(vectors))


def test_feature_table_keeps_only_l_and_h_rows():
    labels = pd.DataFrame({"Shot": [7], "Begin_Time": [0], "End_Time": [300], "Mode_L0_D1_H2": [2]})
    rec = FakeRecord(7, [100, 200, 400], [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    df = feature_table([rec], labels, prefix="f")
    assert df.columns.tolist() == ["shot", "time", "label", "f0", "f1"]
    assert df["time"].tolist() == [100, 200]
    assert df["label"].tolist() == [1, 1]


def test_feature_table_skips_records_with_no_features():
    labels = pd.DataFrame({"Shot": [7], "Begin_Time": [0], "End_Time": [300], "Mode_L0_D1_H2": [0]})
    rec = FakeRecord(7, [], [])
    assert feature_table([rec], labels, prefix="f").empty


def test_ensemble_rows_are_the_inner_join_on_shot_and_time():
    ece = pd.DataFrame({"shot": [1, 1], "time": [100, 200], "label": [0, 1], "e0": [0.1, 0.2]})
    pr = pd.DataFrame({"shot": [1, 2], "time": [200, 200], "label": [1, 0], "p0": [9.0, 8.0]})
    both = ensemble_rows(ece, pr)
    assert both[["shot", "time", "label", "e0", "p0"]].values.tolist() == [[1, 200, 1, 0.2, 9.0]]
