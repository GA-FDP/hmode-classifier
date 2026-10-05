"""CMF plumbing shared by the stages.

Toksearch pipelines are recorded with toksearch_cmf.CmfRun. Stages that are
not toksearch pipelines (training, the ensemble combination) use CmfStage,
which drives cmflib directly and records inputs, outputs and metrics against
one execution.
"""
import filecmp
import os
import shutil
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


def dvc_mirror(src_dir, name):
    """A DVC-trackable copy of a git-tracked directory, under data/mirror/<name>.

    cmflib hands every logged path to `dvc add`, which refuses files git
    already tracks. The copy has the same content, so the hash CMF records is
    the hash of the original.
    """
    dest = os.path.join(ROOT, "data", "mirror", name)
    os.makedirs(dest, exist_ok=True)
    for entry in sorted(os.listdir(src_dir)):
        src = os.path.join(src_dir, entry)
        dst = os.path.join(dest, entry)
        if os.path.isfile(src) and (not os.path.exists(dst) or not filecmp.cmp(src, dst, shallow=False)):
            shutil.copyfile(src, dst)
    return dest
