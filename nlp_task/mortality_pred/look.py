import csv
import json
import os
import random
import numpy as np

with open("/home/peter/files/PyHealth/nlp_task/readmission_pred/mimic4/readmission_pred_data.json", "r") as f:
    data = json.load(f)

count_1 = 0
count_2 = 0

label1 = []
label2 = []



# for i in range(1, len(data)):
#     if data[i]["patient_id"] == data[i-1]["patient_id"]:
#         count_1 += 1
#         if data[i]["discharge_status"] == 0 and data[i-1]["discharge_status"] == 1:
#             print(data[i]["patient_id"])
#             count_2 += 1

for index in range(len(data)):
    if data[index]["label"] == 1:
        count_1 += 1
        label1.append(index)
    else:
        count_2 += 1
        label2.append(index)

print(count_1)
print(count_2)
