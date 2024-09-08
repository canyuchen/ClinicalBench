import json
import csv
patient_dic = {}
with open('/home/peter/files/PyHealth/nlp_task/length_pred/length_pred_data.json', 'r') as f:
    load_samples = json.load(f)
    for sample in load_samples:
        list_info = []
        list_info.append(sample['age'])
        list_info.append(sample['gender'])
        list_info.append(sample['ethnicity'])
        if patient_dic.get(sample['patient_id']) is None:
            patient_dic[sample['patient_id']] = list_info

with open('/home/peter/files/PyHealth/nlp_task/length_pred/patient_dic.json', 'w') as f:
    json.dump(patient_dic, f)
