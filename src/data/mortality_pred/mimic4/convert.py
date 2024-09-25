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

    with open('mortality_pred_data.json', 'r') as f:
        load_samples = json.load(f)
        count = 0
        ICL_str = "_ICL" if ICL else ""
        with open(f'mortality_pred_data{ICL_str}.csv', 'w', newline='') as csvfile:
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
                sentence_0 = "Patient information:\nAge: 37\nGender: female\nConditions: Post-term pregnancy, Other maternal infectious and parasitic diseases complicating pregnancy, third trimester, 41 weeks gestation of pregnancy, Single live birth, Streptococcus, group B, as the cause of diseases classified elsewhere, Second degree perineal laceration during delivery\nProcedures: Introduction of Other Therapeutic Substance into Female Reproductive, Via Natural or Artificial Opening, Delivery of Products of Conception, External Approach, Repair Perineum Muscle, Open Approach\nUsing Drugs: antacids, urologicals, throat preparations, antiinflammatory and antirheumatic products, non-steroids, agents for treatment of hemorrhoids and anal fissures for topical use, antipruritics, incl. antihistamines, anesthetics, etc., opioids, drugs for constipation, bacterial and viral vaccines, combined, calcium, viral vaccines\nWill the patient die because of the above situation?\nAnswer 1 if yes, 0 if no. Answer with only the number.\nAnswer: 0\n\nPatient information:\nAge: 52\nGender: male\nConditions: Hepatic encephalopathy, Defibrination syndrome, Subendocardial infarction, initial episode of care, Esophageal varices in diseases classified elsewhere, with bleeding, Acute kidney failure with lesion of tubular necrosis, Other ascites, Alcoholic cirrhosis of liver, Acute alcoholic intoxication in alcoholism, continuous, Acute alcoholic hepatitis, Other specified disorders of circulatory system\nProcedures: Percutaneous abdominal drainage, Intra-abdominal venous shunt, Venous catheterization, not elsewhere classified\nUsing Drugs: i.v. solution additives, hypothalamic hormones, drugs for peptic ulcer and gastro-oesophageal reflux disease (gord), other nutrients, posterior pituitary lobe hormones, anesthetics, general, insulins and analogues, cardiac stimulants excl. cardiac glycosides, other beta-lactam antibacterials, blood and related products, vitamin k and other hemostatics, other mineral supplements, other diagnostic agents, antiinfectives and antiseptics, excl. combinations with corticosteroids, vitamin b12 and folic acid, vitamin b1, plain and in combination with vitamin b6 and b12, hypnotics and sedatives\nWill the patient die because of the above situation?\nAnswer 1 if yes, 0 if no. Answer with only the number.\nAnswer: 1\n\n"
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
                sentence6 = 'Will the patient die because of the above situation?\n'
                sentence7 = 'Answer 1 if yes, 0 if no. Answer with only the number.\n'
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