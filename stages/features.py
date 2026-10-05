#!/usr/bin/env python
"""Feature rows for every labelled shot, recorded in CMF.

Runs the ECE and PR feature pipelines (the inference code with the ONNX step
skipped) over every shot in the label sheet, joins the hand labels, and writes
one parquet table per diagnostic. Two CMF runs, one per pipeline, each with
the label sheet as a declared input.

    pixi run features                # all 301 labelled shots
    pixi run features --limit 5      # trial
"""
import argparse
import os
import time

import numpy as np
from toksearch_cmf import CmfRun

from hmode_classifier import ECE_Classifier, MODEL_DIR, PR_Classifier
from hmode_classifier.cmf import CMF_PIPELINE, ROOT
from hmode_classifier.features import feature_table
from hmode_classifier.labels import read_labels

LABELS = os.path.join(ROOT, "Regression_Test_Data", "Labels.xlsx")
OUTDIR = os.path.join(ROOT, "data", "features")
TIMES = np.arange(100, 10000, 100)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, help="trial run on the first N labelled shots")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    labels = read_labels(LABELS)
    shots = sorted(labels["Shot"].unique().tolist())[: args.limit]
    suffix = f"-trial{args.limit}" if args.limit else ""
    os.makedirs(OUTDIR, exist_ok=True)

    for name, module, prefix in (("ece", ECE_Classifier, "e"), ("pr", PR_Classifier, "p")):
        run = CmfRun(CMF_PIPELINE, stage=f"features-{name}{suffix}", work_dir=ROOT, inputs=[LABELS])
        clock = time.time()
        records = module.Collect_Labels(MODEL_DIR, shots, TIMES, 1, retain_feature_vector=True,
                                        provenance=run, num_workers=args.workers, classify=False)
        table = feature_table(records, labels, prefix)
        out = os.path.join(OUTDIR, f"{name}{suffix}.parquet")
        table.to_parquet(out, index=False)
        print(f"{name}: {len(shots)} shots, {table['shot'].nunique()} with features, "
              f"{len(table)} rows, {time.time() - clock:.0f} s -> {out}", flush=True)
        run.output(out)
        run.metrics("rows", {"shots_requested": len(shots), "shots_with_features": int(table["shot"].nunique()),
                             "rows": len(table), "h_rows": int(table["label"].sum()),
                             "t0_ms": int(TIMES[0]), "dt_ms": int(TIMES[1] - TIMES[0]), "t_end_ms": int(TIMES[-1])})
        run.finalize()


if __name__ == "__main__":
    main()
