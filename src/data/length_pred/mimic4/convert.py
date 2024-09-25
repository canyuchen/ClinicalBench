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

    with open(f'length_pred_data.json', 'r') as f:
        load_samples = json.load(f)
        count = 0
        ICL_str = "_ICL" if ICL else ""
        with open(f'length_pred_data{ICL_str}.csv', 'w', newline='') as csvfile:
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


                sentence_0 = "Patient information:\nAge: 39\nGender: female\nConditions: Endometriosis of ovary, Endometriosis of pelvic peritoneum, Leiomyoma of uterus, unspecified, Other specified diseases of blood and blood-forming organs\nProcedures: Other unilateral salpingo-oophorectomy, Other local excision or destruction of ovary, Other cystoscopy\nUsing Drugs: antidepressants, opioids, antacids, calcium, other analgesics and antipyretics, throat preparations, antiinflammatory and antirheumatic products, non-steroids, antiemetics and antinauseants, viral vaccines, antiinfectives and antiseptics, excl. combinations with corticosteroids\nPredict the number of weeks of stay in hospital.\nAnswer 1 if no more than one week,\nAnswer 2 if more than one week but not more than two weeks,\nAnswer 3 if more than two weeks.\nAnswer with only the number. Answer: 1\n\nPatient information:\nAge: 38\nGender: male\nConditions: Other displaced fracture of fourth cervical vertebra, initial encounter for closed fracture, Respiratory failure, unspecified, unspecified whether with hypoxia or hypercapnia, Contusion of lung, unilateral, initial encounter, Unspecified injury at unspecified level of cervical spinal cord, initial encounter, Unspecified fracture of sternum, initial encounter for closed fracture, Alcohol abuse with intoxication delirium, Enterocolitis due to Clostridium difficile, not specified as recurrent, Cerebrospinal fluid leak, Wedge compression fracture of second thoracic vertebra, initial encounter for closed fracture, Other displaced fracture of fifth cervical vertebra, initial encounter for closed fracture, Other displaced fracture of sixth cervical vertebra, initial encounter for closed fracture, Other displaced fracture of seventh cervical vertebra, initial encounter for closed fracture, Sprain of unspecified ligament of unspecified ankle, initial encounter, Fall (on) (from) unspecified stairs and steps, initial encounter, Other specified places as the place of occurrence of the external cause, Laceration without foreign body of scalp, initial encounter, Nicotine dependence, cigarettes, uncomplicated, Cannabis use, unspecified, uncomplicated, Anemia, unspecified, Other specified dorsopathies, cervical region\nProcedures: Fusion of 2 or more Cervical Vertebral Joints with Autologous Tissue Substitute, Anterior Approach, Anterior Column, Open Approach, Insertion of Endotracheal Airway into Trachea, Via Natural or Artificial Opening, Release Cervical Spinal Cord, Open Approach, Excision of Cervical Vertebra, Open Approach, Fusion of Cervicothoracic Vertebral Joint with Autologous Tissue Substitute, Posterior Approach, Anterior Column, Open Approach, Reposition Cervical Vertebral Joint, Open Approach, Repair Spinal Meninges, Open Approach, Respiratory Ventilation, 24-96 Consecutive Hours, Introduction of Nutritional Substance into Lower GI, Via Natural or Artificial Opening\nUsing Drugs: other nutrients, insulins and analogues, anxiolytics, opioids, glycogenolytic hormones, other analgesics and antipyretics, agents for treatment of hemorrhoids and anal fissures for topical use, antipruritics, incl. antihistamines, anesthetics, etc., i.v. solution additives, drugs for peptic ulcer and gastro-oesophageal reflux disease (gord), other mineral supplements, other diagnostic agents, antiinfectives and antiseptics, excl. combinations with corticosteroids, viral vaccines, antithrombotic agents, vitamin b12 and folic acid, vitamin b1, plain and in combination with vitamin b6 and b12, antiemetics and antinauseants, other beta-lactam antibacterials, antiepileptics, drugs for constipation, i.v. solutions, antivaricose therapy, anesthetics, general, corticosteroids for systemic use, plain, stomatological preparations, throat preparations, blood and related products, drugs used in addictive disorders, digestives, incl. enzymes, antiobesity preparations, excl. diet products, urologicals, thyroid preparations, antifungals for topical use, other dermatological preparations, other antidiarrheals, antiadrenergic agents, centrally acting, low-ceiling diuretics, thiazides, antihypertensives and diuretics in combination, beta-lactam antibacterials, penicillins, vitamin a and d, incl. combinations of the two, calcium, antihistamines for systemic use, iron preparations, antiinfectives, potassium, lipid modifying agents, plain, other ophthalmologicals, other systemic drugs for obstructive airway diseases, all other therapeutic products, antidiarrheal microorganisms, capillary stabilizing agents, antipsoriatics for systemic use, corticosteroids, plain, corticosteroids, other combinations, other respiratory system products, anesthetics, local, hypnotics and sedatives, medicated dressings, cardiac glycosides, antimigraine preparations, vasodilators used in cardiac diseases, intestinal adsorbents, gonadotropins and other ovulation stimulants, belladonna and derivatives, plain, antispasmodics in combination with psycholeptics, irrigating solutions, low-ceiling diuretics, excl. thiazides, high-ceiling diuretics, antiseptics and disinfectants, other antibacterials, antipsychotics, adrenergics, inhalants, agents against amoebiasis and other protozoal diseases\nPredict the number of weeks of stay in hospital.\nAnswer 1 if no more than one week,\nAnswer 2 if more than one week but not more than two weeks,\nAnswer 3 if more than two weeks.\nAnswer with only the number. Answer: 2\n\nPatient information:\nAge: 57\nGender: male\nConditions: Acute myeloblastic leukemia, not having achieved remission, Organ-limited amyloidosis, Agranulocytosis secondary to cancer chemotherapy, Oral mucositis (ulcerative), unspecified, Other disorders of skin and subcutaneous tissue in diseases classified elsewhere, Gastro-esophageal reflux disease without esophagitis, Pleurodynia, Dizziness and giddiness, Dysuria, Diarrhea, unspecified, Adverse effect of antineoplastic and immunosuppressive drugs, initial encounter, Fever presenting with conditions classified elsewhere, Unspecified place in hospital as the place of occurrence of the external cause\nProcedures: Extraction of Iliac Bone Marrow, Percutaneous Approach, Diagnostic, Introduction of Other Antineoplastic into Central Vein, Percutaneous Approach, Excision of Left Lower Leg Skin, External Approach, Diagnostic\nUsing Drugs: i.v. solution additives, drugs for constipation, antigout preparations, other analgesics and antipyretics, other antineoplastic agents, antiarrhythmics, class i and iii, antithrombotic agents, potassium, other mineral supplements, other diagnostic agents, i.v. solutions, quinolone antibacterials, direct acting antivirals, antiemetics and antinauseants, corticosteroids for systemic use, plain, cytotoxic antibiotics and related substances, antimetabolites, anxiolytics, antacids, calcium, urologicals, antivaricose therapy, drugs for peptic ulcer and gastro-oesophageal reflux disease (gord), antimycotics for systemic use, opioids, other beta-lactam antibacterials, other antibacterials, agents for treatment of hemorrhoids and anal fissures for topical use, antipruritics, incl. antihistamines, anesthetics, etc.\nPredict the number of weeks of stay in hospital.\nAnswer 1 if no more than one week,\nAnswer 2 if more than one week but not more than two weeks,\nAnswer 3 if more than two weeks.\nAnswer with only the number. Answer: 3\n\n"


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
                if ICL:
                    sentence = sentence_0 + sentence0 + sentence1 + sentence2 + sentence3 + sentence4 + sentence5 + sentence6 + sentence7 + sentence8
                else:
                    sentence = sentence0 + sentence1 + sentence2 + sentence3 + sentence4 + sentence5 + sentence6 + sentence7 + sentence8
                writer.writerow({'ID': count, 'VISIT_ID': visit_id, 'SUBJECT_ID': subject_id, 'QUESTION': sentence, 'ANSWER': label})
                count += 1
        print(count)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='path/to/mimic4/dataset', help='Path to mimic4 dataset')
    parser.add_argument('--ICL', type=int, default=False, help='ICL data or not')
    args = parser.parse_args()
    main(args.dataset_path, args.ICL)