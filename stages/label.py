#!/usr/bin/env python
"""Label shots with a chosen weight set, every step recorded in CMF.

Three executions: label-ece and label-pr are toksearch pipelines recorded
with CmfRun, each declaring the weight files it scored with as inputs;
label-ensemble combines them (a CmfStage) and writes one parquet table with
every model's label and H-mode probability per (shot, time).

    pixi run label --shots 202981 202982 --name demo            # v1 weights
    pixi run label --shots-file shots.txt --weights models/v2 --name v2-demo
"""
import argparse
import os
import time

import numpy as np
import pandas as pd
from toksearch_cmf import CmfRun

from hmode_classifier import ECE_Classifier, Ensemble_Classifier, MODEL_DIR, PR_Classifier
from hmode_classifier.cmf import CMF_PIPELINE, ROOT, CmfStage

WEIGHT_FILES = ["ECE_GradBoost_v1.onnx", "ECE_Kmeans_v1.onnx", "PR_GradBoost_v1.onnx", "PR_Kmeans_v1.onnx",
                "Ave_Dist_ECE_v1.npy", "Ave_Dist_PR_v1.npy", "Cluster_loc_ECE_v1.npy", "Cluster_loc_PR_v1.npy"]


def labels_frame(ece_results, pr_results, ensemble_results):
    rows = []
    for model, results in (("ece", ece_results), ("pr", pr_results)):
        for rec in results:
            for t, lab, prob in zip(rec["Label_Times"], rec["LH_Label"], rec["LH_Prob"]):
                rows.append((model, int(rec["shot"]), int(t), int(lab), float(prob[1])))
    for shot, d in ensemble_results.items():
        for t, lab, prob in zip(d["Time"], d["Ensemble_Label"], d["Ensemble_Hmode_Prob"]):
            rows.append(("ensemble", int(shot), int(t), int(lab), float(prob)))
    return pd.DataFrame(rows, columns=["model", "shot", "time", "label", "h_prob"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shots", type=int, nargs="*", default=[])
    parser.add_argument("--shots-file", help="one shot number per line")
    parser.add_argument("--name", required=True, help="output directory name under data/labels/")
    parser.add_argument("--weights", default=str(MODEL_DIR), help="directory with the eight weight files")
    parser.add_argument("--t0", type=int, default=100)
    parser.add_argument("--t1", type=int, default=10000)
    parser.add_argument("--dt", type=int, default=100)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    shots = list(args.shots)
    if args.shots_file:
        with open(args.shots_file) as f:
            shots += [int(line) for line in f if line.strip()]
    shots = sorted(set(shots))
    times = np.arange(args.t0, args.t1, args.dt)
    weights = os.path.abspath(args.weights)
    weight_paths = [os.path.join(weights, name) for name in WEIGHT_FILES]
    outdir = os.path.join(ROOT, "data", "labels", args.name)
    os.makedirs(outdir, exist_ok=True)
    props = {"weights_dir": os.path.relpath(weights, ROOT), "t0_ms": args.t0, "t1_ms": args.t1, "dt_ms": args.dt}

    results = {}
    run_ids = {}
    for name, module in (("ece", ECE_Classifier), ("pr", PR_Classifier)):
        run = CmfRun(CMF_PIPELINE, stage=f"label-{name}", work_dir=ROOT, inputs=weight_paths)
        clock = time.time()
        results[name] = module.Collect_Labels(weights, shots, times, 1, retain_feature_vector=True,
                                              provenance=run, num_workers=args.workers)
        print(f"{name}: {len(shots)} shots in {time.time() - clock:.0f} s", flush=True)
        run.metrics("grid", {k: v for k, v in props.items() if k != "weights_dir"})
        run.finalize()
        run_ids[name] = run.run_id

    stage = CmfStage("label-ensemble", properties={**props, "parent_run_ece": run_ids["ece"],
                                                    "parent_run_pr": run_ids["pr"]})
    for path in weight_paths:
        stage.input(path)
    M = Ensemble_Classifier.load_models(weights)
    dd = Ensemble_Classifier.DataDictionaryBuilder(results["ece"], results["pr"], shots)
    ensemble = Ensemble_Classifier.BuildEnsembleDictionary(
        dd, M["kmeans_ece"], M["kmeans_pr"], M["ave_ece"], M["cluster_ece"], M["ave_pr"], M["cluster_pr"])

    frame = labels_frame(results["ece"], results["pr"], ensemble)
    out = os.path.join(outdir, "labels.parquet")
    frame.to_parquet(out, index=False)
    counts = frame.groupby("model").size().to_dict()
    print(f"{out}: {counts}", flush=True)
    stage.output(out, weights_version=os.path.basename(weights))
    stage.metrics("rows", {f"{k}_rows": int(v) for k, v in counts.items()})
    stage.finalize()


if __name__ == "__main__":
    main()
