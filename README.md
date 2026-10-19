# Hmode-Classifier-For-FDP
This is a classification tool designed to utilize ECE and PR diganostic data. There exists methods for using both individuall or as a combined Ensemble model.

## Install and run

Needs [pixi](https://pixi.sh) and an FDP bearer token (`~/.fdp/token`).

```bash
git clone <repo>
cd hmode-classifier
pixi install
pixi run test              # unit tests, no network
pixi run test-live         # the 20-shot regression against the shipped weights, about 3 min
```

Label shots with the shipped v1 weights:

```python
import numpy as np
from hmode_classifier import Hmode_Ml_Predictions

hm = Hmode_Ml_Predictions("Ensemble", [202981, 202982], np.arange(100, 10000, 100))
results = hm.Predict()      # {'Ensemble': {shot: ...}, 'ECE': [records], 'PR': [records]}
```

Run scripts under `python -m fdp run python your_script.py` so the FDP environment is set.
The Ensemble model on 300 shots at 100 ms steps takes about an hour; the ECE pipeline
dominates, so give `Collect_Labels` more workers on a large machine. The time grid is
free: times outside a shot's data are skipped, and the nearest sample is used otherwise.
Skipped times and the reason are reported in each record's `Labeling_Errors`.

## Provenance

Every stage is recorded in the CMF store in `mlmd/`, with artifacts hashed by DVC. A
label run traces to the weight files it scored with; v2 weights trace further, through
the training run and the feature runs, to the label sheet and to the FDP archive
snapshot each feature run read (shot list, signal specs and archive catalog are in
`cmf_runs/<run>/inputs.json`).

| Stage | Reads | Writes |
| --- | --- | --- |
| `features-ece`, `features-pr` | label sheet, FDP archive | `data/features/{ece,pr}.parquet` |
| `train` | the feature tables, label sheet | `models/v2/` (eight weight files, `split.json`) |
| `label-ece`, `label-pr`, `label-ensemble` | a weights directory, FDP archive | `data/labels/<name>/labels.parquet` |

```bash
pixi run features --workers 32                    # 301 labelled shots; ECE is the slow half
pixi run train                                    # seconds
pixi run label --shots 202981 --name demo --weights models/v2
pixi run python tools/lineage.py                  # prints the chain back to Labels.xlsx
```

cmflib checks out a branch named `mlmd` and commits its pointer files there on every
run; commit your own work on `mlmd` while running stages and fast-forward `main` to it
afterwards. The DVC remote is set in `.dvc/config`.

## Labels

`Regression_Test_Data/Labels.xlsx` is the hand-labelled sheet the models were trained on:
301 shots from the 2024 campaign, one row per L-mode, dither or H-mode window. It is free
for anyone to use, for retraining or otherwise.

The ECE chord radii in `Calc_R` use the radiometer channel frequencies from the
[DIII-D ECE radiometer page](https://fusionga.sharepoint.com/sites/DIII-DDiagnostics/SitePages/ECE-Radiometer.aspx)
and R0 = 1.6955 m (M. Austin, personal communication).

## Weights

`hmode_classifier/Models/` holds the v1 weights the papers report; the regression test
and the default `Predict()` use them. `models/v2/` is produced by `pixi run train` from
the same feature code the inference path runs, following the author's recipe: a seeded
80/20 split by shot for the held-out accuracy (`split.json`, recorded as metrics), then a
refit on every row for the exported weights, as v1 was. A full-data fit agrees with v1 on
99.6% of ECE rows and 99.8% of PR rows; the rest is data reprocessed since v1 was trained.
Averaged over 20 random splits the retrained ensemble scores 99.1% +/- 0.5% on held-out
shots, matching the 99.2% +/- 0.6% in the original training notebook.

The ECE Model:  
![ECE RBF Flow Chart.pdf](https://github.com/user-attachments/files/28619657/ECE.RBF.Flow.Chart.pdf)


The PR Model:  
![PR_Flow_Chart.pdf](https://github.com/user-attachments/files/28619678/PR_Flow_Chart.pdf)


The Ensemble Model:  
![Ensemble Model image.pdf](https://github.com/user-attachments/files/28619672/Ensemble.Model.image.pdf)


Result of Ensemble Model on Randomized Shot Data:  
<img width="422" height="222" alt="Screenshot 2026-06-04 at 6 19 57 PM" src="https://github.com/user-attachments/assets/5eedeca8-6f3d-4b4b-8441-1952994ee871" />

Version Control\
Successful Regression Tests were conducted using the following software environment:\
Python - 3.11.15
NumPy - 1.26.4\
Scikit-Learn - 1.9.0\
ONNXRuntime - 1.26.0\
toksearch - 2.8.0\
toksearch_d3d - 0.9.8

Version Guidance\
Note that only the Regression Test version has been tested to work, while we expect that these versions are compatible, they haven't been rigorously tested\
Python - Version 3.11-3.13 reccomended\
NumPy - Version >= 1.26.0 reccomended\
Scikit-Learn - Version >= 1.9 reccomended\
ONNXRuntime - must support ONNX opset 17\
toksearch/toksearch_d3d - so long as the data is still able to be pulled and the formatting is consistent, the tool is expected to work with any future version (Should version control matter, please report this issue to initiate an update this toolkit to be compatible with the most up to date version of toksearch/FDP)

References:  
Clark, Randall, et al. "Plasma confinement state classification in fusion power plants: Profile reflectometer and ensemble diagnostics." Nuclear Fusion (2026).  
Clark, Randall, et al. "Plasma confinement state classification via FPP relevant microwave diagnostics." Plasma Physics and Controlled Fusion 68.1 (2026): 015022.
