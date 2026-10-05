# H-mode Classifier: CMF Provenance and Retraining Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Package Randall Clark's H-mode classifier so that inference runs, and a rerun of its training, are recorded in CMF with full lineage from the label sheet and the FDP archive to the labels a user gets.

**Architecture:** The four existing modules move unchanged into an installable package `hmode_classifier`; the only code edits are two keyword arguments that let a caller pass a toksearch provenance object and skip the ONNX step. Three stage scripts under `stages/` drive CMF: `features` (CmfRun over the toksearch pipelines, writes per-time feature rows joined to the hand labels), `train` (a non-toksearch CmfStage that fits and exports v2 weights), `label` (two CmfRuns plus a CmfStage that scores shots with a chosen weight set). Training uses the same feature code as inference, which the deprecated repo did not.

**Tech Stack:** Python 3.11, pixi, fdp-core-latest >= 1.24.0 (toksearch 2.17.1, toksearch_d3d 0.14.1, toksearch_cmf 0.2.1, cmflib, dvc), onnxruntime, scikit-learn, skl2onnx, pyarrow, pytest.

**Repository state at start:** `repos/hmode-classifier` is a clone of `RClarkPhysics/Hmode-Classifier-For-FDP` at `30e28f0`. Remote `upstream` is GitHub with pushes disabled; remote `origin` is a local bare repo at `/fusion/projects/dt/sammuli/fdp_dev/scratch/hmode-classifier/origin.git`. cmflib runs `git checkout -B mlmd` on every `Cmf` init, so **commit on `mlmd` once stages start running and fast-forward `main` afterwards**; before that, work on `main`.

**Conventions:**
- Run everything through pixi from the repo root: `pixi run <task>` or `pixi run python -m fdp run python <script>`. Set `RATTLER_CACHE_DIR` and `PIXI_CACHE_DIR` to a path with space (not `/tmp`); `/local-scratch` is fine on this host.
- Tests that need the network are marked `live` and deselected by default (`pytest -m "not live"`); unit tests must pass without a bearer token.
- Commit after every task with a one-line message in the imperative.

---

## File structure

| Path | Responsibility |
| --- | --- |
| `pyproject.toml` | setuptools package `hmode_classifier` with `Models/` as package data; pins |
| `pixi.toml` | the environment and the `test`, `features`, `train`, `label` tasks |
| `hmode_classifier/__init__.py` | exports `Hmode_Ml_Predictions`, `MODEL_DIR` |
| `hmode_classifier/Hmode_ML_Classifier.py` | moved; gains `models_dir=` |
| `hmode_classifier/ECE_Classifier.py` | moved; `Collect_Labels` gains `provenance=`, `num_workers=`, `classify=` |
| `hmode_classifier/PR_Classifier.py` | moved; same three kwargs |
| `hmode_classifier/Ensemble_Classifier.py` | moved; gains `load_models(model_dir)` |
| `hmode_classifier/Models/` | moved; v1 weights, unchanged |
| `hmode_classifier/labels.py` | reads `Labels.xlsx`; mode per (shot, time) with the notebook's boundary rules |
| `hmode_classifier/cmf.py` | `CMF_PIPELINE`, `ROOT`, `CmfStage` for non-toksearch stages |
| `hmode_classifier/features.py` | pure functions: records + labels -> feature DataFrames |
| `hmode_classifier/train.py` | pure functions: DataFrames -> fitted models -> ONNX bytes + stats |
| `stages/features.py` | CmfRun stage: label sheet -> `data/features/{ece,pr}.parquet` |
| `stages/train.py` | CmfStage: features -> `models/v2/` + metrics |
| `stages/label.py` | CmfRun x2 + CmfStage: shots -> `data/labels/<name>/` |
| `tests/` | unit tests (no network) and one `live` regression test |

---

### Task 1: Environment and packaging

**Files:**
- Create: `pyproject.toml`, `pixi.toml`, `.gitignore`, `hmode_classifier/__init__.py`
- Move: `Hmode_ML_Classifier.py`, `ECE_Classifier.py`, `PR_Classifier.py`, `Ensemble_Classifier.py`, `Models/` -> `hmode_classifier/`
- Modify: `hmode_classifier/Hmode_ML_Classifier.py` (MODEL_DIR resolution), the first cell of the four notebooks (import path)
- Test: `tests/test_package.py`

- [ ] **Step 1: Move the modules into the package**

```bash
mkdir -p hmode_classifier tests
git mv Hmode_ML_Classifier.py ECE_Classifier.py PR_Classifier.py Ensemble_Classifier.py Models hmode_classifier/
```

- [ ] **Step 2: Write `hmode_classifier/__init__.py`**

```python
"""H-mode classifier for DIII-D over the Fusion Data Platform.

Randall Clark's ECE, profile-reflectometer and ensemble L/H-mode classifiers
(Nucl. Fusion 2026; PPCF 68 015022), packaged with pretrained weights.
"""
from pathlib import Path

from .Hmode_ML_Classifier import Hmode_Ml_Predictions

#: The v1 weights shipped with the package.
MODEL_DIR = Path(__file__).resolve().parent / "Models"

__all__ = ["Hmode_Ml_Predictions", "MODEL_DIR"]
```

- [ ] **Step 3: Make `Hmode_ML_Classifier` accept a weights directory**

In `hmode_classifier/Hmode_ML_Classifier.py`, change the constructor signature and the `MODEL_DIR` block:

```python
    def __init__(self, Model_Choice, ShotRange, Times, Exclude_Non_Plasma = 1, models_dir = None):
```

and replace

```python
        try:
            self.MODEL_DIR = Path(__file__).resolve().parent / "Models/"
        except NameError:
            self.MODEL_DIR = Path.cwd() / "Models/"
```

with

```python
        if models_dir is None:
            self.MODEL_DIR = Path(__file__).resolve().parent / "Models"
        else:
            self.MODEL_DIR = Path(models_dir).resolve()
```

Add to the docstring, after `Exclude_Non_Plasma`:

```
        models_dir : PATH, optional
            Directory holding the ONNX models and .npy statistics. Defaults to the v1 weights shipped in the package.
```

- [ ] **Step 4: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "hmode-classifier"
version = "0.1.0"
description = "ECE and profile-reflectometer L/H-mode classifier for DIII-D over the Fusion Data Platform"
readme = "README.md"
requires-python = ">=3.11"
license = {text = "MIT"}
authors = [{name = "Randall Clark"}]
dependencies = [
    "numpy>=1.26",
    "scikit-learn>=1.9",
    "onnxruntime>=1.17",
    "toksearch>=2.17",
    "toksearch_d3d>=0.14",
]

[project.optional-dependencies]
cmf = ["toksearch_cmf>=0.2.1", "skl2onnx>=1.17", "pyarrow", "openpyxl", "pandas"]

[tool.setuptools.packages.find]
include = ["hmode_classifier*"]

[tool.setuptools.package-data]
hmode_classifier = ["Models/*.onnx", "Models/*.npy"]

[tool.pytest.ini_options]
testpaths = ["tests"]
markers = ["live: needs the FDP origin and a bearer token"]
addopts = "-m 'not live'"
```

- [ ] **Step 5: Write `pixi.toml`**

```toml
[workspace]
name = "hmode-classifier"
channels = ["conda-forge", "ga-fdp"]
platforms = ["linux-64"]

[dependencies]
python = "3.11.*"
fdp-core-latest = ">=1.24.0"
onnxruntime = "*"
scikit-learn = ">=1.9"
skl2onnx = "*"
openpyxl = "*"
pyarrow = "*"
pandas = "*"
pytest = "*"

[pypi-dependencies]
hmode-classifier = { path = ".", editable = true }

[activation.env]
PYTHONNOUSERSITE = "1"

[tasks]
test = "pytest"
test-live = "python -m fdp run pytest -m live -s"
features = "python -m fdp run python stages/features.py"
train = "python stages/train.py"
label = "python -m fdp run python stages/label.py"
```

- [ ] **Step 6: Write `.gitignore`**

```
.pixi/
__pycache__/
*.pyc
*.egg-info/
/data/
/models/v2/
/cmf_runs/
```

Note: `mlmd` and `cmf_artifacts` are committed by cmflib itself on the `mlmd` branch; leave them out of `.gitignore`.

- [ ] **Step 7: Update the notebooks' import**

```bash
for nb in "Regression Tests.ipynb" "Testing Out ECE.ipynb" "Testing Out Ens.ipynb" "Testing Out PR.ipynb"; do
  sed -i 's/"import Hmode_ML_Classifier/"from hmode_classifier import Hmode_ML_Classifier/' "$nb"
done
grep -c "from hmode_classifier import Hmode_ML_Classifier" "Regression Tests.ipynb"
```

Expected: `2` (two import cells in that notebook), and `1` in each of the others.

- [ ] **Step 8: Write the failing package test**

`tests/test_package.py`:

```python
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
```

- [ ] **Step 9: Install the environment and run the test**

```bash
export RATTLER_CACHE_DIR=/local-scratch/sammuli/rattler-cache PIXI_CACHE_DIR=/local-scratch/sammuli/pixi-cache
pixi install
pixi run test
```

Expected: `3 passed`.

- [ ] **Step 10: Commit**

```bash
git add -A
git commit -m "Package the classifier as hmode_classifier with the v1 weights as package data"
```

---

### Task 2: Provenance and feature-only hooks in the classifiers

**Files:**
- Modify: `hmode_classifier/ECE_Classifier.py`, `hmode_classifier/PR_Classifier.py`, `hmode_classifier/Ensemble_Classifier.py`
- Test: `tests/test_hooks.py`

- [ ] **Step 1: Write the failing test**

`tests/test_hooks.py`:

```python
import inspect

from hmode_classifier import ECE_Classifier, Ensemble_Classifier, PR_Classifier


def test_collect_labels_accepts_provenance_and_feature_only_mode():
    for module in (ECE_Classifier, PR_Classifier):
        params = inspect.signature(module.Collect_Labels).parameters
        assert params["provenance"].default is None
        assert params["num_workers"].default is None
        assert params["classify"].default is True


def test_ensemble_models_load_from_a_directory():
    from hmode_classifier import MODEL_DIR
    models = Ensemble_Classifier.load_models(MODEL_DIR)
    assert set(models) == {"kmeans_ece", "kmeans_pr", "ave_ece", "cluster_ece", "ave_pr", "cluster_pr"}
    assert models["cluster_ece"].shape == (200, 7)
    assert models["cluster_pr"].shape == (200, 10)
```

- [ ] **Step 2: Run it to see it fail**

```bash
pixi run test tests/test_hooks.py
```

Expected: FAIL, `KeyError: 'provenance'` and `AttributeError: ... has no attribute 'load_models'`.

- [ ] **Step 3: Edit `ECE_Classifier.Collect_Labels`**

Signature:

```python
def Collect_Labels(MODEL_DIR,ShotRange, Times, Exclude_Non_Plasma,retain_feature_vector = False,
                   provenance = None, num_workers = None, classify = True):
```

Inside `CalcLabels`, replace the block from `#Perform the Labelling using the ONNX Model` through the `else:` that sets `record['Label_Times']` with:

```python
            Coef = np.array([v['Weights'] for v in DD_Coef.values()])
            if Coef.size == 0 or not classify:
                record['LH_Label'] = []
                record['LH_Prob'] = []
                record['Label_Times'] = np.array([v['Time'] for v in DD_Coef.values()])
            else:
                #Perform the Labelling using the ONNX Model
                ONNX_ECE = ort.InferenceSession(MODEL_DIR/"ECE_GradBoost_v1.onnx")
                input_name = ONNX_ECE.get_inputs()[0].name
                output_names = [out.name for out in ONNX_ECE.get_outputs()]
                outputs = ONNX_ECE.run(output_names, {input_name: Coef.astype(np.float32)})
                record['LH_Label'] = outputs[0]
                record['LH_Prob'] = outputs[1]
                record['Label_Times'] = np.array([v['Time'] for v in DD_Coef.values()])
```

Note `Label_Times` is now set in both branches so a feature-only run still reports which times produced a feature vector. Replace the final compute line with:

```python
    records = pipeline.compute_multiprocessing(num_workers=num_workers, provenance=provenance)
```

- [ ] **Step 4: Edit `PR_Classifier.Collect_Labels` the same way**

Signature:

```python
def Collect_Labels(MODEL_DIR,ShotRange,Times,Exclude_Non_Plasma,retain_feature_vector = False,
                   provenance = None, num_workers = None, classify = True):
```

Replace from `#Perform the Labelling using the ONNX Model` through the `else:` block with:

```python
            if Spline_Values.size == 0 or not classify:
                record['LH_Label'] = []
                record['LH_Prob'] = []
                record['Label_Times'] = np.array([v['Time'] for v in DD_Spline.values()])
            else:
                #Perform the Labelling using the ONNX Model
                ONNX_PR = ort.InferenceSession(MODEL_DIR/"PR_GradBoost_v1.onnx")
                input_name = ONNX_PR.get_inputs()[0].name
                output_names = [out.name for out in ONNX_PR.get_outputs()]
                outputs = ONNX_PR.run(output_names, {input_name: Spline_Values.astype(np.float32)})
                record['LH_Label'] = outputs[0]
                record['LH_Prob'] = outputs[1]
                record['Label_Times'] = np.array([v['Time'] for v in DD_Spline.values()])
```

and the compute line with:

```python
    records = pipeline.compute_multiprocessing(num_workers=num_workers, provenance=provenance)
```

- [ ] **Step 5: Add `load_models` to `Ensemble_Classifier` and use it**

After the `PR_CUTOFF = 2.5` line add:

```python
def load_models(MODEL_DIR):
    """The ensemble's k-means sessions and training statistics from one directory."""
    MODEL_DIR = Path(MODEL_DIR)
    return {
        'kmeans_ece': ort.InferenceSession(MODEL_DIR/"ECE_Kmeans_v1.onnx"),
        'kmeans_pr': ort.InferenceSession(MODEL_DIR/"PR_Kmeans_v1.onnx"),
        'ave_ece': np.load(MODEL_DIR/'Ave_Dist_ECE_v1.npy'),
        'cluster_ece': np.load(MODEL_DIR/'Cluster_loc_ECE_v1.npy'),
        'ave_pr': np.load(MODEL_DIR/'Ave_Dist_PR_v1.npy'),
        'cluster_pr': np.load(MODEL_DIR/'Cluster_loc_PR_v1.npy'),
    }
```

Add `from pathlib import Path` to the imports. In `Collect_Labels`, replace the six `ONNX_Kmeans_ECE = ...` through `Cluster_PR = ...` lines with:

```python
    M = load_models(MODEL_DIR)
```

and the `BuildEnsembleDictionary(DD,ONNX_Kmeans_ECE,...)` call with:

```python
    Ensemble_Result = BuildEnsembleDictionary(DD,M['kmeans_ece'],M['kmeans_pr'],M['ave_ece'],M['cluster_ece'],M['ave_pr'],M['cluster_pr'])
```

The v2 weights in Task 5 keep the `_v1` file names so this loader works for both; the directory, not the file name, identifies the version.

- [ ] **Step 6: Run the tests**

```bash
pixi run test
```

Expected: all pass (5 tests).

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Let callers pass a provenance object, a worker count and a feature-only flag"
```

---

### Task 3: Labels and feature tables

**Files:**
- Create: `hmode_classifier/labels.py`, `hmode_classifier/features.py`
- Test: `tests/test_labels.py`, `tests/test_features.py`

- [ ] **Step 1: Write the failing label test**

`tests/test_labels.py`:

```python
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
    # end of an L window is L; start of an H window is H; past the last window is -1
    assert modes.tolist() == [0, 0, 1, 2, 2, -1, -1]


def test_unknown_shot_is_unlabelled(tmp_path):
    labels = read_labels(_sheet(tmp_path))
    assert modes_for(labels, 2, np.array([100])).tolist() == [-1]
```

- [ ] **Step 2: Write `hmode_classifier/labels.py`**

```python
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
```

- [ ] **Step 3: Run the label tests**

```bash
pixi run test tests/test_labels.py
```

Expected: `2 passed`.

- [ ] **Step 4: Write the failing feature-table test**

`tests/test_features.py`:

```python
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
```

- [ ] **Step 5: Write `hmode_classifier/features.py`**

```python
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
```

- [ ] **Step 6: Run the feature tests**

```bash
pixi run test tests/test_features.py
```

Expected: `3 passed`.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "Add label lookup and feature tables for training"
```

---

### Task 4: CMF helpers and the features stage

**Files:**
- Create: `hmode_classifier/cmf.py`, `stages/features.py`, `stages/__init__.py` (empty)
- Modify: none
- Test: `tests/test_cmf_stage.py`

- [ ] **Step 1: Initialise DVC with a local remote**

```bash
mkdir -p /fusion/projects/dt/sammuli/fdp_dev/scratch/hmode-classifier/dvc-remote
pixi run dvc init -q
pixi run dvc remote add -d local /fusion/projects/dt/sammuli/fdp_dev/scratch/hmode-classifier/dvc-remote
git add .dvc .dvcignore
git commit -m "Initialise DVC with a local remote"
```

- [ ] **Step 2: Write `hmode_classifier/cmf.py`**

```python
"""CMF plumbing shared by the stages.

Toksearch pipelines are recorded with toksearch_cmf.CmfRun. Stages that are
not toksearch pipelines (training, the ensemble combination) use CmfStage,
which drives cmflib directly and records inputs, outputs and metrics against
one execution.
"""
import os
import sys
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CMF_PIPELINE = "hmode-classifier"


class CmfStage:
    def __init__(self, stage, properties=None):
        from cmflib.cmf import Cmf

        self.run_id = uuid.uuid4().hex
        self.cmf = Cmf(filepath=os.path.join(ROOT, "mlmd"), pipeline_name=CMF_PIPELINE)
        self.cmf.create_context(pipeline_stage=stage)
        self.cmf.create_execution(
            execution_type=stage,
            custom_properties={"run": self.run_id,
                               "code.script": os.path.relpath(sys.argv[0], ROOT),
                               **(properties or {})},
        )

    def _log(self, path, event, props):
        rel = os.path.relpath(os.path.abspath(path), ROOT)
        self.cmf.log_dataset(rel, event, custom_properties=dict(props))

    def input(self, path, **props):
        self._log(path, "INPUT", props)

    def output(self, path, **props):
        self._log(path, "OUTPUT", props)

    def metrics(self, name, values):
        self.cmf.log_execution_metrics(name, dict(values))

    def finalize(self):
        self.cmf.finalize()
```

- [ ] **Step 3: Write the stage-helper test**

`tests/test_cmf_stage.py`:

```python
import os

from hmode_classifier import cmf


def test_root_is_the_repository():
    assert os.path.isfile(os.path.join(cmf.ROOT, "pyproject.toml"))


def test_pipeline_name_is_stable():
    assert cmf.CMF_PIPELINE == "hmode-classifier"
```

Run `pixi run test tests/test_cmf_stage.py`; expected `2 passed`.

- [ ] **Step 4: Write `stages/features.py`**

```python
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
        run = CmfRun(CMF_PIPELINE, stage=f"features-{name}{suffix}", work_dir=ROOT, inputs=[LABELS],
                     properties={"t0_ms": 100, "dt_ms": 100, "t_end_ms": 10000})
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
                             "rows": len(table), "h_rows": int(table["label"].sum())})
        run.finalize()


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Commit the stage, then run a trial on the `mlmd` branch**

cmflib checks out `mlmd` on its first run; commit first so the script's commit is recorded.

```bash
git add -A
git commit -m "Add the CMF features stage"
export RATTLER_CACHE_DIR=/local-scratch/sammuli/rattler-cache PIXI_CACHE_DIR=/local-scratch/sammuli/pixi-cache
pixi run features --limit 5
git branch --show-current
```

Expected: two lines like `ece: 5 shots, 5 with features, N rows, ~40 s -> data/features/ece-trial5.parquet`, and the branch is now `mlmd`.

- [ ] **Step 6: Check the trial table**

```bash
pixi run python -c "
import pandas as pd
for n in ('ece','pr'):
    df = pd.read_parquet(f'data/features/{n}-trial5.parquet'); print(n, df.shape, df.label.value_counts().to_dict(), df.columns.tolist()[:5])"
```

Expected: ece has columns `shot, time, label, e0..e6` (7 features), pr has `p0..p9` (10 features), both labels present.

- [ ] **Step 7: Run the full stage**

```bash
pixi run features 2>&1 | tail -3
```

Expected: about 300 shots each, roughly 5 minutes per pipeline, writing `data/features/ece.parquet` and `data/features/pr.parquet`. Check that `cmf_runs/` holds two new run directories and `data/features/ece.parquet.dvc` exists.

- [ ] **Step 8: Commit on `mlmd` and fast-forward `main`**

```bash
git add -A
git commit -m "Record the features runs"
git checkout -q main && git merge -q --ff-only mlmd && git checkout -q mlmd
git log --oneline -3
```

---

### Task 5: Training library and stage

**Files:**
- Create: `hmode_classifier/train.py`, `stages/train.py`
- Test: `tests/test_train.py`

- [ ] **Step 1: Write the failing training test**

`tests/test_train.py`:

```python
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
```

- [ ] **Step 2: Run it to see it fail**

```bash
pixi run test tests/test_train.py
```

Expected: FAIL with `ModuleNotFoundError: No module named 'hmode_classifier.train'`.

- [ ] **Step 3: Write `hmode_classifier/train.py`**

```python
"""Fit and export the classifiers.

Hyperparameters are the ones in the original training notebook: a histogram
gradient-boosted classifier (learning rate 0.1, 100 iterations, 31 leaves, no
L2) per diagnostic, and a 200-cluster k-means per diagnostic whose cluster
centres and mean in-cluster distance weight the ensemble.
"""
import random

import numpy as np
from skl2onnx import to_onnx as _to_onnx
from skl2onnx.common.data_types import FloatTensorType
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier

GRADBOOST = dict(learning_rate=0.1, max_iter=100, max_leaf_nodes=31, l2_regularization=0.0)
N_CLUSTERS = 200
ONNX_OPSET = 17


def feature_columns(df):
    return [c for c in df.columns if c not in ("shot", "time", "label")]


def split_shots(shots, fraction=0.8, seed=0):
    """Train/test shot lists; the same split for the same seed."""
    shots = sorted(int(s) for s in shots)
    rng = random.Random(seed)
    train = sorted(rng.sample(shots, int(fraction * len(shots))))
    test = [s for s in shots if s not in set(train)]
    return train, test


def fit_classifier(df):
    x = df[feature_columns(df)].to_numpy(np.float32)
    return HistGradientBoostingClassifier(**GRADBOOST).fit(x, df["label"].to_numpy())


def fit_clusters(df, n_clusters=N_CLUSTERS, seed=0):
    """k-means on the training rows, with the statistics the ensemble needs."""
    x = df[feature_columns(df)].to_numpy(np.float32)
    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto").fit(x)
    centres = km.cluster_centers_.astype(np.float64)
    distances = np.linalg.norm(x - centres[km.labels_], axis=1)
    return km, centres, float(np.mean(distances))


def to_onnx(model, width):
    """ONNX bytes with the same input/output names and shapes as the v1 files."""
    onx = _to_onnx(model, initial_types=[("input", FloatTensorType([None, width]))], target_opset=ONNX_OPSET)
    return onx.SerializeToString()


def accuracy(clf, df):
    x = df[feature_columns(df)].to_numpy(np.float32)
    return float(np.mean(clf.predict(x) == df["label"].to_numpy()))
```

- [ ] **Step 4: Run the tests**

```bash
pixi run test tests/test_train.py
```

Expected: `3 passed`. If `to_onnx` of the KMeans emits a `scores` output named differently from v1, the test on `cluster` (output 0) still holds; the ensemble code only reads output 0.

- [ ] **Step 5: Write `stages/train.py`**

```python
#!/usr/bin/env python
"""Train v2 weights from the feature tables, recorded in CMF.

Inputs: data/features/ece.parquet and data/features/pr.parquet (from
stages/features.py) and the label sheet they were joined with. Outputs: the
eight files the package's Ensemble loader expects, under models/v2/, plus
train/test accuracy per model and the ensemble accuracy on held-out shots.

The shot split is seeded and written to models/v2/split.json. One shared
split: the ECE and PR models are each trained on all of their own rows for
the training shots, and the ensemble is scored on held-out shots that have
both.

    pixi run train
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

from hmode_classifier import Ensemble_Classifier
from hmode_classifier.cmf import ROOT, CmfStage
from hmode_classifier.features import ensemble_rows
from hmode_classifier.train import (N_CLUSTERS, accuracy, feature_columns, fit_classifier,
                                    fit_clusters, split_shots, to_onnx)

FEATURES = os.path.join(ROOT, "data", "features")
LABELS = os.path.join(ROOT, "Regression_Test_Data", "Labels.xlsx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--fraction", type=float, default=0.8)
    parser.add_argument("--out", default=os.path.join(ROOT, "models", "v2"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    stage = CmfStage("train", properties={"seed": args.seed, "train_fraction": args.fraction,
                                          "n_clusters": N_CLUSTERS, "weights_version": "v2"})
    tables = {name: pd.read_parquet(os.path.join(FEATURES, f"{name}.parquet")) for name in ("ece", "pr")}
    for name in tables:
        stage.input(os.path.join(FEATURES, f"{name}.parquet"))
    stage.input(LABELS)

    shots = sorted(set(tables["ece"]["shot"]) | set(tables["pr"]["shot"]))
    train_shots, test_shots = split_shots(shots, args.fraction, args.seed)
    with open(os.path.join(args.out, "split.json"), "w") as f:
        json.dump({"seed": args.seed, "fraction": args.fraction, "train": train_shots, "test": test_shots}, f)

    metrics = {}
    fitted = {}
    for name, tag in (("ece", "ECE"), ("pr", "PR")):
        df = tables[name]
        train = df[df["shot"].isin(train_shots)]
        test = df[df["shot"].isin(test_shots)]
        clf = fit_classifier(train)
        km, centres, mean_dist = fit_clusters(train)
        width = len(feature_columns(df))
        with open(os.path.join(args.out, f"{tag}_GradBoost_v1.onnx"), "wb") as f:
            f.write(to_onnx(clf, width))
        with open(os.path.join(args.out, f"{tag}_Kmeans_v1.onnx"), "wb") as f:
            f.write(to_onnx(km, width))
        np.save(os.path.join(args.out, f"Cluster_loc_{tag}_v1.npy"), centres)
        np.save(os.path.join(args.out, f"Ave_Dist_{tag}_v1.npy"), np.float64(mean_dist))
        fitted[name] = clf
        metrics.update({f"{name}_train_rows": len(train), f"{name}_test_rows": len(test),
                        f"{name}_train_accuracy": accuracy(clf, train), f"{name}_test_accuracy": accuracy(clf, test)})
        print(f"{name}: train {len(train)} rows acc {metrics[f'{name}_train_accuracy']:.4f}, "
              f"test {len(test)} rows acc {metrics[f'{name}_test_accuracy']:.4f}", flush=True)

    # Ensemble accuracy on held-out shots that have both feature sets.
    both = ensemble_rows(tables["ece"], tables["pr"])
    both = both[both["shot"].isin(test_shots)]
    M = Ensemble_Classifier.load_models(args.out)
    xe = both[feature_columns(tables["ece"])].to_numpy(np.float32)
    xp = both[feature_columns(tables["pr"])].to_numpy(np.float32)
    pe = fitted["ece"].predict_proba(xe)[:, 1]
    pp = fitted["pr"].predict_proba(xp)[:, 1]
    ce = M["kmeans_ece"].run(None, {"input": xe})[0]
    cp = M["kmeans_pr"].run(None, {"input": xp})[0]
    de = np.linalg.norm(xe - M["cluster_ece"][ce], axis=1)
    dp = np.linalg.norm(xp - M["cluster_pr"][cp], axis=1)
    _, label = Ensemble_Classifier.Ensemble_Model(pe, pp, M["ave_ece"], M["ave_pr"], de, dp)
    metrics["ensemble_test_rows"] = len(both)
    metrics["ensemble_test_accuracy"] = float(np.mean(label == both["label"].to_numpy()))
    print(f"ensemble: test {len(both)} rows acc {metrics['ensemble_test_accuracy']:.4f}", flush=True)

    stage.output(args.out, weights_version="v2")
    stage.metrics("accuracy", metrics)
    stage.finalize()


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Commit and run**

```bash
git add -A
git commit -m "Add the training library and the CMF train stage"
pixi run train
ls models/v2
```

Expected: accuracies in the high 90s for all three (the paper reports about 99% on held-out shots), and `models/v2/` holds the eight weight files plus `split.json`. The notebook's reported ensemble test accuracy was 99.2% with std 0.6% over 100 random splits; one split in the 98 to 100% range is consistent.

- [ ] **Step 7: Commit the run and fast-forward main**

```bash
git add -A
git commit -m "Record the v2 training run"
git checkout -q main && git merge -q --ff-only mlmd && git checkout -q mlmd
```

---

### Task 6: The label stage (inference with provenance)

**Files:**
- Create: `stages/label.py`
- Test: `tests/test_label_stage.py`

- [ ] **Step 1: Write the failing test for the result writer**

`tests/test_label_stage.py`:

```python
import numpy as np
import pandas as pd

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
```

- [ ] **Step 2: Run it to see it fail**

```bash
pixi run test tests/test_label_stage.py
```

Expected: FAIL, `ModuleNotFoundError: No module named 'stages.label'`.

- [ ] **Step 3: Write `stages/label.py`**

```python
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
        run = CmfRun(CMF_PIPELINE, stage=f"label-{name}", work_dir=ROOT, inputs=weight_paths, properties=props)
        clock = time.time()
        results[name] = module.Collect_Labels(weights, shots, times, 1, retain_feature_vector=True,
                                              provenance=run, num_workers=args.workers)
        print(f"{name}: {len(shots)} shots in {time.time() - clock:.0f} s", flush=True)
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
```

- [ ] **Step 4: Run the unit test, commit, then a live trial**

```bash
pixi run test
git add -A
git commit -m "Add the CMF label stage"
pixi run label --shots 202981 202982 --name trial-v1
pixi run label --shots 202981 202982 --name trial-v2 --weights models/v2
```

Expected: each run prints three lines and writes `data/labels/trial-v1/labels.parquet`; the second scores with v2 weights. Then:

```bash
git add -A && git commit -m "Record the label trials" && git checkout -q main && git merge -q --ff-only mlmd && git checkout -q mlmd
```

---

### Task 7: Regression test as pytest

**Files:**
- Create: `tests/test_regression_live.py`
- Modify: none

- [ ] **Step 1: Write the test**

```python
"""Randall Clark's 20-shot regression, against the shipped v1 weights.

Live: needs the FDP origin and a bearer token. Run with `pixi run test-live`.
"""
import pickle

import numpy as np
import pytest

from hmode_classifier import Hmode_Ml_Predictions
from hmode_classifier.cmf import ROOT

SHOTS = [200410, 200939, 201183, 201454, 202029, 202033, 202138, 203011, 203013, 203036,
         203045, 203047, 203092, 203110, 203271, 203297, 204049, 204181, 204461, 204463]
TIMES = np.arange(100, 10000, 100)
# The reference for this shot was made before its data was recalculated; the
# regression notebook notes the same thing.
KNOWN_RECALCULATED = {203045}
REFERENCE = {"ECE": "ECE_Regression_Dict.pkl", "PR": "PR_Regression_Dict.pkl", "Ensemble": "Ens_Regression_Dict.pkl"}


def _reference(name):
    with open(f"{ROOT}/Regression_Test_Data/{REFERENCE[name]}", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def results():
    return Hmode_Ml_Predictions("Ensemble", SHOTS, TIMES).Predict()


def _probabilities(results, model, i, shot):
    if model == "Ensemble":
        d = results[model][shot]
        return np.asarray(d["Time"]), np.asarray(d["Ensemble_Hmode_Prob"])
    rec = results[model][i]
    probs = [p[1] for p in rec["LH_Prob"]] if len(rec["LH_Prob"]) else []
    return np.asarray(rec["Label_Times"]), np.asarray(probs)


@pytest.mark.live
@pytest.mark.parametrize("model", ["ECE", "PR", "Ensemble"])
def test_no_label_flips_against_the_reference(results, model):
    ref = _reference(model)
    flips = {}
    for i, shot in enumerate(SHOTS):
        if shot in KNOWN_RECALCULATED:
            continue
        times, probs = _probabilities(results, model, i, shot)
        common, ri, fi = np.intersect1d(np.asarray(ref[shot]["Time"]), times, return_indices=True)
        if common.size == 0:
            continue
        n = int(np.sum(np.round(np.asarray(ref[shot]["H_prob"])[ri]) != np.round(probs[fi])))
        if n:
            flips[shot] = n
    assert flips == {}, f"{model} label flips vs reference: {flips}"
```

- [ ] **Step 2: Run it live**

```bash
pixi run test-live 2>&1 | tail -5
```

Expected: `3 passed` in about 3 minutes.

- [ ] **Step 3: Commit**

```bash
git add -A
git commit -m "Turn the regression notebook into a live pytest"
git checkout -q main && git merge -q --ff-only mlmd && git checkout -q mlmd
```

---

### Task 8: Lineage check and README

**Files:**
- Create: `tools/lineage.py`
- Modify: `README.md`

- [ ] **Step 1: Write `tools/lineage.py`**

A script that opens the mlmd store and prints, for the newest `label-ensemble` execution, the chain back to the label sheet: the weights directory it scored with, the `train` execution that produced those weights (if v2), the `features-*` executions behind that, and the label sheet artifact. It fails with exit code 1 if a v2 label run does not trace to `Labels.xlsx`.

```python
#!/usr/bin/env python
"""Print the lineage of the newest label-ensemble run; exit 1 if v2 weights do not trace to the label sheet.

    pixi run python tools/lineage.py
"""
import os
import sys

from cmflib import cmfquery

from hmode_classifier.cmf import CMF_PIPELINE, ROOT


def main():
    q = cmfquery.CmfQuery(os.path.join(ROOT, "mlmd"))
    stages = q.get_pipeline_stages(CMF_PIPELINE)
    print("stages:", stages)
    execs = q.get_all_executions_in_stage("label-ensemble")
    if execs.empty:
        print("no label-ensemble execution yet"); return 1
    latest = execs.sort_values("id").iloc[-1]
    weights = latest.get("custom_properties_weights_dir", "")
    print(f"label-ensemble execution {latest['id']}: weights {weights}")
    arts = q.get_all_artifacts_for_execution(int(latest["id"]))
    for _, a in arts.iterrows():
        print(f"  {a['event']:<7} {a['name']}")
    if "v2" not in str(weights):
        print("v1 weights: lineage ends at the shipped files (expected)"); return 0
    train = q.get_all_executions_in_stage("train")
    feats = [s for s in stages if s.startswith("features-")]
    ok = (not train.empty) and feats
    names = set()
    for stage in feats:
        for _, e in q.get_all_executions_in_stage(stage).iterrows():
            names |= set(q.get_all_artifacts_for_execution(int(e["id"]))["name"])
    ok = ok and any("Labels.xlsx" in n for n in names)
    print("traces to Labels.xlsx:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
```

If `cmfquery` method names differ in the installed cmflib (check with `pixi run python -c "from cmflib import cmfquery; help(cmfquery.CmfQuery)"`), use the equivalents it lists; the three calls needed are: stages of a pipeline, executions in a stage, artifacts of an execution.

- [ ] **Step 2: Run it after the v2 label trial**

```bash
pixi run python tools/lineage.py; echo exit=$?
```

Expected: prints the artifact list and `traces to Labels.xlsx: True`, exit 0.

- [ ] **Step 3: Rewrite `README.md`**

Keep Randall's description, flow charts, references and version table. Replace the usage paragraph with:

````markdown
## Install and run

```bash
git clone <repo>
cd hmode-classifier
pixi install
pixi run test              # unit tests, no network
```

Label shots with the shipped v1 weights (needs an FDP bearer token):

```python
import numpy as np
from hmode_classifier import Hmode_Ml_Predictions
hm = Hmode_Ml_Predictions("Ensemble", [202981, 202982], np.arange(100, 10000, 100))
results = hm.Predict()
```

Run it under `python -m fdp run python your_script.py` so the FDP environment is set.

## Provenance

Every recorded run goes into the CMF store in `mlmd/`:

| Stage | Reads | Writes |
| --- | --- | --- |
| `features-ece`, `features-pr` | label sheet, FDP archive | `data/features/{ece,pr}.parquet` |
| `train` | the feature tables, label sheet | `models/v2/` (eight weight files, `split.json`) |
| `label-ece`, `label-pr`, `label-ensemble` | weight files, FDP archive | `data/labels/<name>/labels.parquet` |

```bash
pixi run features                                 # ~10 min, 301 labelled shots
pixi run train                                    # seconds
pixi run label --shots 202981 --name demo --weights models/v2
pixi run python tools/lineage.py                  # prints the chain back to Labels.xlsx
```

cmflib commits its metadata on a branch named `mlmd`; fast-forward `main` to it after a run.

## Weights

`hmode_classifier/Models/` holds the v1 weights the papers report. `models/v2/` is produced by `pixi run train` from the same feature code the inference path uses, with a recorded seed and shot split; it is not byte-identical to v1 because the v1 split was not recorded.
````

- [ ] **Step 4: Run everything once more and commit**

```bash
pixi run test
git add -A
git commit -m "Add a lineage check and document the provenance workflow"
git checkout -q main && git merge -q --ff-only mlmd && git checkout -q mlmd
git push -q origin main mlmd
```

---

## Self-review

- Spec coverage: packaging (T1), provenance hooks (T2), label join + feature tables (T3), features stage with CmfRun (T4), retraining with recorded split and ONNX export (T5), inference with provenance and parent links (T6), regression as pytest (T7), lineage check and docs (T8). The "v1 stays, v2 is the first lineaged version" decision is implemented by `--weights` defaulting to v1 and `models/v2` being a separate output.
- Placeholders: none; the one conditional ("if cmfquery method names differ") names the exact three calls needed.
- Types: `Collect_Labels(..., provenance, num_workers, classify)` is used identically in T4 and T6; `load_models` keys match between T2, T5 and T6; `feature_table(records, labels, prefix)` and `ensemble_rows(ece, pr)` match between T3, T4 and T5; `labels_frame` signature matches T6's test.
