from pyhealth.datasets import MIMIC3Dataset

mimic3_ds = MIMIC3Dataset(
        root="/home/peter/files/PyHealth/data/mimiciii",
        tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        dev=False,
)
from pyhealth.tasks import readmission_prediction_mimic3_fn

mimic3_ds = mimic3_ds.set_task(task_fn=readmission_prediction_mimic3_fn)
samples = mimic3_ds.samples