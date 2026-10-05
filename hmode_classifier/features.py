"""Feature rows for training, from the same code that extracts them at inference."""
import numpy as np
import pandas as pd

from .labels import modes_for


def feature_table(records, labels, prefix):
    """One row per (shot, time) with an L or H label; dither and unlabelled times dropped.

    `records` come from ECE_Classifier/PR_Classifier.Collect_Labels with
    retain_feature_vector=True. The label column is 0 for L and 1 for H, the
    encoding the classifiers were trained with.
    """
    frames = []
    for rec in records:
        times = np.asarray(rec["Label_Times"])
        vectors = np.asarray(rec["Feature_Vector"])
        if times.size == 0 or vectors.size == 0:
            continue
        modes = modes_for(labels, rec["shot"], times)
        keep = np.isin(modes, (0, 2))
        if not keep.any():
            continue
        df = pd.DataFrame(vectors[keep], columns=[f"{prefix}{i}" for i in range(vectors.shape[1])])
        df.insert(0, "label", (modes[keep] // 2).astype(int))
        df.insert(0, "time", times[keep].astype(int))
        df.insert(0, "shot", int(rec["shot"]))
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["shot", "time", "label"])
    return pd.concat(frames, ignore_index=True)


def ensemble_rows(ece, pr):
    """Rows where both feature sets exist for the same (shot, time)."""
    return ece.merge(pr.drop(columns=["label"]), on=["shot", "time"], how="inner")
