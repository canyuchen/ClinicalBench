import csv
import json
import argparse

def main(dataset_path, ICL):
    icd9_to_long_prcedures = {}
    icd9_to_long_diagnoses = {}
    atc_to_drug = {}
    with open('../../ATC_dic.json', 'r') as f:
        atc_to_drug = json.load(f)

    with open(f'{dataset_path}/d_icd_procedures.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            icd9_to_long_prcedures[row['icd_code']] = row['long_title']

    with open(f'{dataset_path}/d_icd_diagnoses.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            icd9_to_long_diagnoses[row['icd_code']] = row['long_title']    


    with open('readmission_pred_data.json', 'r') as f:
        load_samples = json.load(f)
        count = 0
        ICL_str = "_ICL" if ICL else ""
        with open(f'readmission_pred_data{ICL_str}.csv', 'w', newline='') as csvfile:
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
                            drugs[i][j] = drugs[i][j].lower()
                        except:
                            drugs[i][j] = 'Unknown Drug'
                sentence_0 = "Patient information:\nAge: 28\nGender: female\nConditions: First-degree perineal laceration, delivered, with or without mention of antepartum condition, Motorcycle driver injured in collision with fixed or stationary object in nontraffic accident\nProcedures: Repair of other current obstetric laceration\nUsing Drugs: drugs for constipation, agents for treatment of hemorrhoids and anal fissures for topical use, antipruritics, incl. antihistamines, anesthetics, etc., antacids, calcium, opioids, other analgesics and antipyretics, throat preparations, antiinflammatory and antirheumatic products, non-steroids, cough suppressants, excl. combinations with expectorants\nWill the patient be readmitted to the hospital within two weeks?\nAnswer 1 for yes, 0 for no. Answer with only the number.\nAnswer: 0\n\nPatient information:\nAge: 44\nGender: male\nConditions: Poisoning by salicylates, Acute kidney failure, unspecified, Acidosis, Suicide and self-inflicted poisoning by analgesics, antipyretics, and antirheumatics\nProcedures: Venous catheterization for renal dialysis, Hemodialysis\nUsing Drugs: other nutrients, i.v. solution additives, intestinal adsorbents, antithrombotic agents, antiarrhythmics, class i and iii, anxiolytics, drugs used in addictive disorders, vitamin b1, plain and in combination with vitamin b6 and b12, other diagnostic agents, vitamin b12 and folic acid, potassium, drugs for constipation\nWill the patient be readmitted to the hospital within two weeks?\nAnswer 1 for yes, 0 for no. Answer with only the number.\nAnswer: 1\n\n"
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
                sentence6 = 'Will the patient be readmitted to the hospital within two weeks?\n'
                sentence7 = 'Answer 1 for yes, 0 for no. Answer with only the number.\n'
                sentence8 = 'Answer: '
                if ICL:
                    sentence = sentence_0 + sentence0 + sentence1 + sentence2 + sentence3 + sentence4 + sentence5 + sentence6 + sentence7 + sentence8
                else:    
                    sentence = sentence0 + sentence1 + sentence2 + sentence3 + sentence4 + sentence5 + sentence6 + sentence7 + sentence8
                writer.writerow({'ID': count, 'VISIT_ID': visit_id, 'SUBJECT_ID': subject_id, 'QUESTION': sentence, 'ANSWER': label})
                count += 1

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--dataset_path', type=str, required=True, help='Path to the mimic4 dataset')
    argparser.add_argument('--ICL', type=int, required=True, help='ICL data or not')
    args = argparser.parse_args()
    main(args.dataset_path, args.ICL)