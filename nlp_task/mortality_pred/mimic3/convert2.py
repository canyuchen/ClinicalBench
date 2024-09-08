import csv
import json
icd9_to_long_prcedures = {}
icd9_to_long_diagnoses = {}
atc_to_drug = {}
with open('/home/peter/files/PyHealth/nlp_task/drug_pred/ATC_dic.json', 'r') as f:
    atc_to_drug = json.load(f)

with open('/home/peter/files/PyHealth/data/mimiciii/D_ICD_PROCEDURES.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        icd9_to_long_prcedures[row['ICD9_CODE']] = row['LONG_TITLE']

with open('/home/peter/files/PyHealth/data/mimiciii/D_ICD_DIAGNOSES.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        icd9_to_long_diagnoses[row['ICD9_CODE']] = row['LONG_TITLE']  

with open('/home/peter/files/PyHealth/nlp_task/mortality_pred/mortality_pred_data.json', 'r') as f:
    load_samples = json.load(f)
    count = 0
    with open('/home/peter/files/PyHealth/nlp_task/mortality_pred/mortality_pred_data_ICL.csv', 'w', newline='') as csvfile:
        fieldnames = ['ID', 'VISIT_ID', 'SUBJECT_ID', 'QUESTION', 'ANSWER']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for sample in load_samples:
            visit_id = sample['visit_id']
            subject_id = sample['patient_id']
            conditions = sample['conditions']
            procedures = sample['procedures']
            age = sample['age']
            gender = sample['gender']
            if gender == 'M':
                gender = 'male'
            elif gender == 'F':
                gender = 'female'
            ethnicity = sample['ethnicity']
            label = sample['label']
            drugs = sample['drugs']
            for i in range(len(conditions)):
                for j in range(len(conditions[i])):
                    try:
                        conditions[i][j] = icd9_to_long_diagnoses[conditions[i][j]]
                    except:
                        conditions[i][j] = 'Unknown Diagnosis'
            for i in range(len(procedures)):
                for j in range(len(procedures[i])):
                    try:
                        procedures[i][j] = icd9_to_long_prcedures[procedures[i][j]]
                    except:
                        procedures[i][j] = 'Unknown Procedure'
            for i in range(len(drugs)):
                for j in range(len(drugs[i])):
                    try:
                        drugs[i][j] = atc_to_drug[drugs[i][j]]
                        # 改成小写
                        drugs[i][j] = drugs[i][j].lower()
                    except:
                        drugs[i][j] = 'Unknown Drug'
            sentence_0 = "Patient information:\nAge: 43\nGender: female\nConditions: Coronary atherosclerosis of native coronary artery, Intermediate coronary syndrome, Diabetes mellitus without mention of complication, type I [juvenile type], not stated as uncontrolled, Unspecified essential hypertension, Pure hypercholesterolemia, Tobacco use disorder\nProcedures: (Aorto)coronary bypass of two coronary arteries, Left heart cardiac catheterization, Extracorporeal circulation auxiliary to open heart surgery, Coronary arteriography using two catheters, Angiocardiography of left heart structures\nUsing Drugs: other analgesics and antipyretics, antipsychotics, vasodilators used in cardiac diseases, antacids, urologicals, anxiolytics, antidepressants, potassium, iron preparations, lipid modifying agents, plain, beta blocking agents, drugs for peptic ulcer and gastro-oesophageal reflux disease (gord), dopaminergic agents, thyroid preparations, opioids, other nutrients, cardiac stimulants excl. cardiac glycosides, drugs for constipation, i.v. solution additives, calcium, propulsives, antiinflammatory and antirheumatic products, non-steroids, other antibacterials, high-ceiling diuretics, antithrombotic agents, other beta-lactam antibacterials, other mineral supplements\nWill the patient die because of the above situation?\nAnswer 1 if yes, 0 if no. Answer with only the number.\nAnswer: 0\n\nPatient information:\nAge: 86\nGender: male\nConditions: Intracerebral hemorrhage, Pneumonitis due to inhalation of food or vomitus, Unspecified essential hypertension, Aortocoronary bypass status, Coronary atherosclerosis of unspecified type of vessel, native or graft\nProcedures: Continuous invasive mechanical ventilation for less than 96 consecutive hours\nUsing Drugs: beta blocking agents, vitamin b1, plain and in combination with vitamin b6 and b12, i.v. solution additives, antiepileptics, quinolone antibacterials, other antibacterials, drugs for peptic ulcer and gastro-oesophageal reflux disease (gord), other mineral supplements, other diagnostic agents, anxiolytics, anesthetics, general, opioids, antiemetics and antinauseants, hypnotics and sedatives\nWill the patient die because of the above situation?\nAnswer 1 if yes, 0 if no. Answer with only the number.\nAnswer: 1\n\n"
            sentence0 = "Patient information:\n"
            sentence1 = "Age: " + str(age) + '\n'
            sentence2 = "Gender: " + str(gender) + '\n'
            sentence3 = 'Conditions: '
            for i in range(len(conditions[0])):
                sentence3 += f'{conditions[0][i]}, '
            sentence3 = sentence3[:-2] + '\n'
            sentence4 = 'Procedures: '
            for i in range(len(procedures[0])):
                sentence4 += f'{procedures[0][i]}, '
            sentence4 = sentence4[:-2] + '\n'
            sentence5 = 'Using Drugs: '
            for i in range(len(drugs[0])):
                sentence5 += f'{drugs[0][i]}, '
            sentence5 = sentence5[:-2] + '\n'
            sentence6 = 'Will the patient die because of the above situation?\n'
            sentence7 = 'Answer 1 if yes, 0 if no. Answer with only the number.\n'
            sentence8 = 'Answer: '
            sentence = sentence_0 + sentence0 + sentence1 + sentence2 + sentence3 + sentence4 + sentence5 + sentence6 + sentence7 + sentence8
            writer.writerow({'ID': count, 'VISIT_ID': visit_id, 'SUBJECT_ID': subject_id, 'QUESTION': sentence, 'ANSWER': label})
            count += 1
