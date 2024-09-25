import csv
import json
import argparse

def main(dataset_path, ICL):
    icd9_to_long_prcedures = {}
    icd9_to_long_diagnoses = {}
    atc_to_drug = {}
    with open('../../ATC_dic.json', 'r') as f:
        atc_to_drug = json.load(f)

    with open(f'{dataset_path}/D_ICD_PROCEDURES.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            icd9_to_long_prcedures[row['ICD9_CODE']] = row['LONG_TITLE']

    with open(f'{dataset_path}/D_ICD_DIAGNOSES.csv', 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            icd9_to_long_diagnoses[row['ICD9_CODE']] = row['LONG_TITLE']  
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
                sentence_0 = "Patient information:\nAge: 31\nGender: female\nConditions: Acute respiratory failure, Myasthenia gravis with (acute) exacerbation, Other specified cardiac dysrhythmias, Diarrhea, Unspecified essential hypertension, Iron deficiency anemia, unspecified\nProcedures: Continuous invasive mechanical ventilation for 96 consecutive hours or more, Insertion of endotracheal tube, Injection or infusion of immunoglobulin, Non-invasive mechanical ventilation, Enteral infusion of concentrated nutritional substances, Transfusion of packed cells\nUsing Drugs: immunosuppressants, corticosteroids for systemic use, plain, drugs for peptic ulcer and gastro-oesophageal reflux disease (gord), other nutrients, i.v. solutions, antivaricose therapy, anticholinergic agents, beta blocking agents, anesthetics, general, parasympathomimetics, other antidiarrheals, calcium, other analgesics and antipyretics, antithrombotic agents, antacids, potassium, i.v. solution additives, sulfonamides and trimethoprim, ace inhibitors, plain, antipropulsives, antidepressants, belladonna and derivatives, plain, anxiolytics, hypnotics and sedatives, other cardiac preparations, antiseptics and disinfectants, antiepileptics\nWill the patient be readmitted to the hospital within two weeks?\nAnswer 1 for yes, 0 for no. Answer with only the number.\nAnswer: 0\n\nPatient information:\nAge: 62\nGender: male\nConditions: Acute myocardial infarction of unspecified site, initial episode of care, Congestive heart failure, unspecified, Unknown Diagnosis, Acute kidney failure with lesion of tubular necrosis, Other and unspecified complications of medical care, not elsewhere classified, Pneumonia, organism unspecified, Late effects of cerebrovascular disease, hemiplegia affecting unspecified side, Unspecified pleural effusion, Unknown Diagnosis, Anticoagulants causing adverse effects in therapeutic use, Diabetes mellitus without mention of complication, type II or unspecified type, not stated as uncontrolled, Anemia, unspecified, Atrial fibrillation, Attention to tracheostomy, Unspecified essential hypertension, Pure hypercholesterolemia, Coronary atherosclerosis of unspecified type of vessel, native or graft, Aortocoronary bypass status\nProcedures: Thoracentesis, Enteral infusion of concentrated nutritional substances, Infusion of drotrecogin alfa (activated), Continuous invasive mechanical ventilation for less than 96 consecutive hours\nUsing Drugs: antithrombotic agents, beta blocking agents, antiarrhythmics, class i and iii, other nutrients, lipid modifying agents, plain, drugs for peptic ulcer and gastro-oesophageal reflux disease (gord), other analgesics and antipyretics, other beta-lactam antibacterials, other antibacterials, i.v. solution additives, hypnotics and sedatives, anesthetics, general, agents against amoebiasis and other protozoal diseases, quinolone antibacterials, ace inhibitors, plain, high-ceiling diuretics, other antidiarrheals, antacids, insulins and analogues, decongestants and other nasal preparations for topical use, intestinal antiinfectives, calcium, antifungals for topical use, anxiolytics, beta-lactam antibacterials, penicillins, adrenergics, inhalants, antidepressants, potassium\nWill the patient be readmitted to the hospital within two weeks?\nAnswer 1 for yes, 0 for no. Answer with only the number.\nAnswer: 1\n\n"
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
    argparser.add_argument('--dataset_path', type=str, required=True, help='Path to the mimic3 dataset')
    argparser.add_argument('--ICL', type=int, required=True, help='ICL data or not')
    args = argparser.parse_args()
    main(args.dataset_path, args.ICL)