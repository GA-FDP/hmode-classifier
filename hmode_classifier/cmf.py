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
