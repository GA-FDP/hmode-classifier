"""Hand labels: L-mode (0), dither (1), H-mode (2) time windows per shot.

The sheet has one row per window: Shot, Begin_Time, End_Time, Mode_L0_D1_H2.
A time is labelled by the window it falls in, half-open [begin, end), with
the regression notebook's two boundary rules: the end of an L window is L,
and the start of an H window is H. Times in no window are -1.
"""
import numpy as np
import pandas as pd

COLUMNS = ["Shot", "Begin_Time", "End_Time", "Mode_L0_D1_H2"]


def read_labels(path):
    df = pd.read_excel(path)[COLUMNS]
    return df.sort_values(["Shot", "Begin_Time"]).reset_index(drop=True)


def modes_for(labels, shot, times):
    """Mode per time for one shot; -1 outside every window."""
    times = np.asarray(times)
    modes = np.full(len(times), -1, dtype=int)
    for _, row in labels[labels["Shot"] == shot].iterrows():
        begin, end, mode = row["Begin_Time"], row["End_Time"], int(row["Mode_L0_D1_H2"])
        modes[(times >= begin) & (times < end)] = mode
        if mode == 0:
            modes[times == end] = 0
        elif mode == 2:
            modes[times == begin] = 2
    return modes
