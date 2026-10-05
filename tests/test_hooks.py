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
