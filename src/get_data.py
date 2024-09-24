from pyhealth.datasets import MIMIC3Dataset, MIMIC4Dataset
from pyhealth.tasks import length_of_stay_prediction_mimic3_fn, mortality_prediction_mimic3_fn, readmission_prediction_mimic3_fn, length_of_stay_prediction_mimic4_fn, mortality_prediction_mimic4_fn, readmission_prediction_mimic4_fn
import json
import argparse


def main(mimc3_path, mimic4_path):
    # mimic3
    if mimc3_path:
        mimic3_ds = MIMIC3Dataset(
            root=mimc3_path,
            tables=["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        )
        breakpoint()
        len_mimic3 = mimic3_ds.set_task(task_fn=length_of_stay_prediction_mimic3_fn)

        with open(f'data/length_pred/mimic3/length_pred_data.json', 'w') as f:
            json.dump(len_mimic3.samples, f, indent=2)
            
        mortality_mimic3 = mimic3_ds.set_task(task_fn=mortality_prediction_mimic3_fn)

        with open(f'data/mortality_pred/mimic3/mortality_pred_data.json', 'w') as f:
            json.dump(mortality_mimic3.samples, f, indent=2)

        readmission_mimic3 = mimic3_ds.set_task(task_fn=readmission_prediction_mimic3_fn)

        with open(f'data/readmission_pred/mimic3/readmission_pred_data.json', 'w') as f:
            json.dump(readmission_mimic3.samples, f, indent=2)

    # mimic4
    if mimic4_path:
        mimic4_ds = MIMIC4Dataset(
            root=mimic4_path,
            tables=["diagnoses_icd", "procedures_icd", "prescriptions"],
            code_mapping={"NDC": ("ATC", {"target_kwargs": {"level": 3}})},
        )

        len_mimic4 = mimic4_ds.set_task(task_fn=length_of_stay_prediction_mimic4_fn)

        with open(f'data/length_pred/mimic4/length_pred_data.json', 'w') as f:
            json.dump(len_mimic4.samples, f, indent=2)
            
        mortality_mimic4 = mimic4_ds.set_task(task_fn=mortality_prediction_mimic4_fn)

        with open(f'data/mortality_pred/mimic4/mortality_pred_data.json', 'w') as f:
            json.dump(mortality_mimic4.samples, f, indent=2)
            
        readmission_mimic4 = mimic4_ds.set_task(task_fn=readmission_prediction_mimic4_fn)

        with open(f'data/readmission_pred/mimic4/readmission_pred_data.json', 'w') as f:
            json.dump(readmission_mimic4.samples, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimic3_path", type=str, required=False)
    parser.add_argument("--mimic4_path", type=str, required=False)
    args = parser.parse_args()

    main(args.mimic3_path, args.mimic4_path)