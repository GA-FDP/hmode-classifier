import os

from hmode_classifier import cmf


def test_root_is_the_repository():
    assert os.path.isfile(os.path.join(cmf.ROOT, "pyproject.toml"))


def test_pipeline_name_is_stable():
    assert cmf.CMF_PIPELINE == "hmode-classifier"
