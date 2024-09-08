import csv
import json
import os
import random
import numpy as np

with open("/home/peter/files/PyHealth/nlp_task/length_pred/mimic4/length_pred_data.json", "r") as f:
    data = json.load(f)

count1 = 0
count2 = 0
count3 = 0

for i in data:
    if i["label"] == 1:
        count1 += 1
    elif i["label"] == 2:
        count2 += 1
    elif i["label"] == 3:
        count3 += 1

print(count1, count2, count3)