import csv
import json
import numpy as np


dataset = 'mimic3'
random_index = 0


with open(f'/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/train_index_{random_index}.npy', 'rb') as f:
    train_index = np.load(f)
train_index = train_index.tolist()
print(len(train_index))
with open(f'/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/val_index_{random_index}.npy', 'rb') as f:
    val_index = np.load(f)
val_index = val_index.tolist()
print(len(val_index))
with open(f'/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/test_index_{random_index}.npy', 'rb') as f:
    test_index = np.load(f)
test_index = test_index.tolist()


val_data = []
train_data = []
test_data = []

with open(f'/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/length_pred_data.csv', 'r') as f:
    csvreader = csv.DictReader(f)
    for idx, row in enumerate(csvreader):
        if row['VISIT_ID'] not in val_index:
            continue    
        input = row['QUESTION']
        input = input[:-215] + '\nAnswer: '
        output = row['ANSWER']
        dic = {'instruction': "Given the patient information, predict the number of weeks of stay in hospital.\nAnswer 1 if no more than one week,\nAnswer 2 if more than one week but not more than two weeks,\nAnswer 3 if more than two weeks.\nAnswer with only the number", 'input': input, 'output': output}
        val_data.append(dic)

with open(f'/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/length_pred_data.csv', 'r') as f:
    csvreader = csv.DictReader(f)  
    for idx, row in enumerate(csvreader):
        if row['VISIT_ID'] not in train_index:
            continue
        input = row['QUESTION']
        input = input[:-215] + '\nAnswer: '
        output = row['ANSWER']
        dic = {'instruction': "Given the patient information, predict the number of weeks of stay in hospital.\nAnswer 1 if no more than one week,\nAnswer 2 if more than one week but not more than two weeks,\nAnswer 3 if more than two weeks.\nAnswer with only the number", 'input': input, 'output': output}
        train_data.append(dic)

with open(f'/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/length_pred_data.csv', 'r') as f:
    csvreader = csv.DictReader(f)
    for idx, row in enumerate(csvreader):
        if row['VISIT_ID'] not in test_index:
            continue
        input = row['QUESTION']
        input = input[:-215] + '\nAnswer: '
        output = row['ANSWER']
        dic = {'instruction': "Given the patient information, predict the number of weeks of stay in hospital.\nAnswer 1 if no more than one week,\nAnswer 2 if more than one week but not more than two weeks,\nAnswer 3 if more than two weeks.\nAnswer with only the number", 'input': input, 'output': output}
        test_data.append(dic)

import random
random.seed(33)
random.shuffle(train_data)
train_data = val_data + train_data
with open('/home/peter/files/PyHealth/nlp_task/length_pred/train_data.json', 'w') as f:
    json.dump(train_data, f)

# with open('/home/peter/files/PyHealth/nlp_task/length_pred/test_data.json', 'w') as f:
#     json.dump(test_data, f)
    
