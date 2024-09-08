import csv
import json
icd9_to_long_prcedures = {}
icd9_to_long_diagnoses = {}
atc_to_drug = {}
with open('/home/peter/files/PyHealth/nlp_task/drug_pred/ATC_dic.json', 'r') as f:
    atc_to_drug = json.load(f)

with open('/home/peter/files/PyHealth/data/mimic-iv2.2/d_icd_procedures.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        icd9_to_long_prcedures[row['icd_code']] = row['long_title']

with open('/home/peter/files/PyHealth/data/mimic-iv2.2/d_icd_diagnoses.csv', 'r', encoding='utf-8') as file:
    reader = csv.DictReader(file)
    for row in reader:
        icd9_to_long_diagnoses[row['icd_code']] = row['long_title']  

with open('/home/peter/files/PyHealth/nlp_task/length_pred/eicu/length_pred_data.json', 'r') as f:
    load_samples = json.load(f)
    count = 0
    with open('/home/peter/files/PyHealth/nlp_task/length_pred/eicu/length_pred_data.csv', 'w', newline='') as csvfile:
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
            # for i in range(len(conditions)):
            #     for j in range(len(conditions[i])):
            #         try:
            #             conditions[i][j] = icd9_to_long_diagnoses[conditions[i][j]]
            #         except:
            #             conditions[i][j] = 'Unknown Diagnosis'
            # for i in range(len(procedures)):
            #     for j in range(len(procedures[i])):
            #         try:
            #             procedures[i][j] = icd9_to_long_prcedures[procedures[i][j]]
            #         except:
            #             procedures[i][j] = 'Unknown Procedure'
            # for i in range(len(drugs)):
            #     for j in range(len(drugs[i])):
            #         try:
            #             drugs[i][j] = atc_to_drug[drugs[i][j]]
            #             # 改成小写
            #             drugs[i][j] = drugs[i][j].lower()
            #         except:
            #             drugs[i][j] = 'Unknown Drug'


            # sentence_0 = "Patient information:\nAge: 74\nGender: male\nConditions: Subendocardial infarction, initial episode of care, Intestinal infection due to Clostridium difficile, Congestive heart failure, unspecified, Coronary atherosclerosis of native coronary artery, Unspecified essential hypertension, Pure hypercholesterolemia, Abdominal aneurysm without mention of rupture, Percutaneous transluminal coronary angioplasty status, Personal history of malignant neoplasm of large intestine\nProcedures: Unknown Procedure, Combined right and left heart cardiac catheterization, Coronary arteriography using two catheters\nUsing Drugs: i.v. solution additives, cardiac stimulants excl. cardiac glycosides, other nutrients, antithrombotic agents, potassium, expectorants, excl. combinations with cough suppressants, drugs for peptic ulcer and gastro-oesophageal reflux disease (gord), hypnotics and sedatives, other analgesics and antipyretics, opioids, lipid modifying agents, plain, belladonna and derivatives, plain, other mineral supplements, other diagnostic agents, agents against amoebiasis and other protozoal diseases, anxiolytics, quinolone antibacterials, ace inhibitors, plain, beta blocking agents, other antibacterials\nPredict the number of weeks of stay in hospital.\nAnswer 1 if no more than one week,\nAnswer 2 if more than one week but not more than two weeks,\nAnswer 3 if more than two weeks.\nAnswer with only the number. Answer: 1\n\nPatient information:\nAge: 36\nGender: male\nConditions: Thoracic aneurysm without mention of rupture, Aortic valve disorders, Congenital insufficiency of aortic valve, Cardiac complications, not elsewhere classified, Other specified cardiac dysrhythmias, Unspecified essential hypertension\nProcedures: Open and other replacement of aortic valve, Resection of vessel with replacement, thoracic vessels, Other operations on vessels of heart, Extracorporeal circulation auxiliary to open heart surgery\nUsing Drugs: drugs for functional gastrointestinal disorders, antiarrhythmics, class i and iii, drugs for constipation, drugs for peptic ulcer and gastro-oesophageal reflux disease (gord), other analgesics and antipyretics, antiinflammatory and antirheumatic products, non-steroids, opioids, other nutrients, anesthetics, general, antiinfectives and antiseptics, excl. combinations with corticosteroids, calcium, arteriolar smooth muscle, agents acting on, cardiac stimulants excl. cardiac glycosides, i.v. solution additives, other beta-lactam antibacterials, insulins and analogues, propulsives, hypnotics and sedatives, other mineral supplements, other diagnostic agents, beta blocking agents, high-ceiling diuretics, potassium, antithrombotic agents, antacids, urologicals, throat preparations, ace inhibitors, plain, angiotensin ii receptor blockers (arbs), plain\nPredict the number of weeks of stay in hospital.\nAnswer 1 if no more than one week,\nAnswer 2 if more than one week but not more than two weeks,\nAnswer 3 if more than two weeks.\nAnswer with only the number. Answer: 2\n\nPatient information:\nAge: 73\nGender: male\nConditions: Closed fracture of shaft of femur, Acute posthemorrhagic anemia, Subendocardial infarction, initial episode of care, Other postoperative infection, Unspecified septicemia, Severe sepsis, Septic shock, Unknown Diagnosis, Cardiogenic shock, Cardiac complications, not elsewhere classified, Paroxysmal ventricular tachycardia, Ventricular fibrillation, Congestive heart failure, unspecified, Atrial fibrillation, Unknown Diagnosis, Pneumonia, organism unspecified, Unspecified fall, Coronary atherosclerosis of native coronary artery, Cardiac pacemaker in situ\nProcedures: Open reduction of fracture with internal fixation, femur, Closed reduction of fracture without internal fixation, femur, Left heart cardiac catheterization, Coronary arteriography using two catheters, Continuous invasive mechanical ventilation for 96 consecutive hours or more, Venous catheterization, not elsewhere classified, Enteral infusion of concentrated nutritional substances\nUsing Drugs: i.v. solution additives, opioids, other analgesics and antipyretics, beta blocking agents, selective calcium channel blockers with direct cardiac effects, antithrombotic agents, vasodilators used in cardiac diseases, agents for treatment of hemorrhoids and anal fissures for topical use, angiotensin ii receptor blockers (arbs), plain, antidepressants, drugs for constipation, drugs for peptic ulcer and gastro-oesophageal reflux disease (gord), anti-dementia drugs, lipid modifying agents, plain, antipsychotics, antipruritics, incl. antihistamines, anesthetics, etc., anxiolytics, antiarrhythmics, class i and iii, quinolone antibacterials, antacids, urologicals, expectorants, excl. combinations with cough suppressants, antiinfectives and antiseptics, excl. combinations with corticosteroids, calcium, other beta-lactam antibacterials, i.v. solutions, antivaricose therapy, other mineral supplements, other diagnostic agents, high-ceiling diuretics, anesthetics, general, muscle relaxants, peripherally acting agents, other antibacterials, other nutrients, hypnotics and sedatives, beta-lactam antibacterials, penicillins, cardiac stimulants excl. cardiac glycosides, anterior pituitary lobe hormones and analogues, other drugs for obstructive airway diseases, inhalants, adrenergics, inhalants, potassium, antifungals for topical use, ace inhibitors, plain\nPredict the number of weeks of stay in hospital.\nAnswer 1 if no more than one week,\nAnswer 2 if more than one week but not more than two weeks,\nAnswer 3 if more than two weeks.\nAnswer with only the number. Answer: 3\n\n"


            sentence0 = "Patient information:\n"
            sentence1 = "Age: " + str(age) + '\n'
            sentence2 = "Gender: " + str(gender) + '\n'
            sentence3 = 'Conditions: '
            for i in range(len(conditions[0])):
                if conditions[0][i] == '':
                    continue
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
            sentence6 = 'Predict the number of weeks of stay in hospital.\n'
            sentence7 = 'Answer 1 if no more than one week,\nAnswer 2 if more than one week but not more than two weeks,\nAnswer 3 if more than two weeks.\n'
            sentence8 = 'Answer with only the number. Answer: '
            sentence = sentence0 + sentence1 + sentence2 + sentence3 + sentence4 + sentence5 + sentence6 + sentence7 + sentence8
            writer.writerow({'ID': count, 'VISIT_ID': visit_id, 'SUBJECT_ID': subject_id, 'QUESTION': sentence, 'ANSWER': label})
            count += 1
    print(count)
