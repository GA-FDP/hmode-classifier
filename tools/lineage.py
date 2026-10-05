#!/usr/bin/env python
"""Print the lineage of a label run; exit 1 if v2 weights do not trace to the label sheet.

    pixi run python tools/lineage.py             # newest label-ensemble run
    pixi run python tools/lineage.py --run <id>  # a specific execution id
"""
import argparse
import os
import sys

from cmflib import cmfquery

from hmode_classifier.cmf import CMF_PIPELINE, ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=int, help="execution id of a label-ensemble run")
    args = parser.parse_args()

    q = cmfquery.CmfQuery(os.path.join(ROOT, "mlmd"))
    runs = q.get_all_executions_in_stage(f"{CMF_PIPELINE}/label-ensemble")
    if runs.empty:
        print("no label-ensemble execution yet")
        return 1
    runs = runs.sort_values("id")
    run = runs[runs["id"] == args.run].iloc[0] if args.run else runs.iloc[-1]
    print(f"label-ensemble execution {run['id']}: {run['Execution']}")
    print(f"  commit {run['Git_End_Commit'][:10]}  weights {run['custom_properties_weights_dir']}")

    arts = q.get_all_artifacts_for_execution(int(run["id"]))
    for _, a in arts.sort_values("event").iterrows():
        print(f"  {a['event']:<7} {a['name']}")

    weights = [n for n in arts["name"] if n.split(":")[0] == run["custom_properties_weights_dir"]]
    if not weights:
        print("weights directory was not recorded as an input")
        return 1
    if "models/" not in weights[0]:
        print("shipped v1 weights: the chain ends at the package files (expected)")
        return 0

    print("parents of", weights[0])
    producer = q.get_producer_execution(weights[0])
    print(f"  produced by execution {producer.id if producer else '?'} "
          f"({producer.properties['Context_Type'].string_value if producer else '?'})")
    parents = q.get_all_parent_artifacts(weights[0])
    names = parents["name"].tolist()
    for n in names:
        print(f"  {n}")
    stages = q.get_all_parent_executions(weights[0])["Context_Type"].tolist()
    print("  via stages:", ", ".join(stages))
    ok = any("Labels.xlsx" in n for n in names) and any("features-" in s for s in stages)
    print("traces to Labels.xlsx through the features stages:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
