import numpy as np
import pandas as pd

from hmode_classifier.labels import modes_for, read_labels


def _sheet(tmp_path):
    df = pd.DataFrame({
        "Shot": [1, 1, 1],
        "Begin_Time": [0, 1750, 1950],
        "End_Time": [1750, 1950, 5000],
        "Mode_L0_D1_H2": [0, 1, 2],
    })
    path = tmp_path / "Labels.xlsx"
    df.to_excel(path, index=False)
    return path


def test_modes_follow_the_notebook_boundary_rules(tmp_path):
    labels = read_labels(_sheet(tmp_path))
    modes = modes_for(labels, 1, np.array([100, 1750, 1800, 1950, 4999, 5000, 6000]))
    # a later window overrides the end-of-L rule at 1750; the start of an H
    # window is H; the end of the last window and beyond are -1
    assert modes.tolist() == [0, 1, 1, 2, 2, -1, -1]


def test_end_of_a_trailing_l_window_is_l(tmp_path):
    df = pd.DataFrame({"Shot": [1], "Begin_Time": [0], "End_Time": [1750], "Mode_L0_D1_H2": [0]})
    path = tmp_path / "Labels.xlsx"
    df.to_excel(path, index=False)
    assert modes_for(read_labels(path), 1, np.array([1750, 1751])).tolist() == [0, -1]


def test_unknown_shot_is_unlabelled(tmp_path):
    labels = read_labels(_sheet(tmp_path))
    assert modes_for(labels, 2, np.array([100])).tolist() == [-1]
