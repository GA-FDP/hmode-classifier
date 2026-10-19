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
# The reference for this shot predates a reprocessing of its reflectometer
# data that moved the pedestal formation by about 10 ms (author, 2026-10).
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
    compared = 0
    for i, shot in enumerate(SHOTS):
        if shot in KNOWN_RECALCULATED:
            continue
        times, probs = _probabilities(results, model, i, shot)
        common, ri, fi = np.intersect1d(np.asarray(ref[shot]["Time"]), times, return_indices=True)
        if common.size == 0:
            continue
        compared += common.size
        n = int(np.sum(np.round(np.asarray(ref[shot]["H_prob"])[ri]) != np.round(probs[fi])))
        if n:
            flips[shot] = n
    assert compared > 300, f"{model}: only {compared} labelled times compared; the fetch probably failed"
    assert flips == {}, f"{model} label flips vs reference: {flips}"
