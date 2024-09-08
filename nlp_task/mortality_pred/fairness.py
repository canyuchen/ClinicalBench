import csv
import json

patient_dic = {}
with open('/home/peter/files/PyHealth/nlp_task/mortality_pred/patient_dic.json', 'r') as f:
    patient_dic = json.load(f)

with open('/home/peter/files/PyHealth/nlp_task/readmission_pred/readmission_pred_result_data_3shot.csv', 'r') as f:
    csvreader = csv.DictReader(f)

    # calculate gender EO
    count_M = 0
    count_M_00 = 0 # predicted 0, actual 0
    count_M_01 = 0 # predicted 0, actual 1
    count_M_10 = 0
    count_M_11 = 0
    count_F = 0
    count_F_00 = 0
    count_F_01 = 0
    count_F_10 = 0
    count_F_11 = 0




    # calculate ethnicity EO
    count_white = 0
    count_black = 0
    count_asian = 0

    count_white_00 = 0
    count_white_01 = 0
    count_white_10 = 0
    count_white_11 = 0

    count_black_00 = 0
    count_black_01 = 0
    count_black_10 = 0
    count_black_11 = 0

    count_asian_00 = 0
    count_asian_01 = 0
    count_asian_10 = 0
    count_asian_11 = 0


    for row in csvreader:
        patient_id = row['SUBJECT_ID']
        if patient_dic[patient_id][1] == 'M':
            count_M += 1
            if row['PREDICTION'] == '0' and row['ANSWER'] == '0':
                count_M_00 += 1
            elif row['PREDICTION'] == '0' and row['ANSWER'] == '1':
                count_M_01 += 1
            elif row['PREDICTION'] == '1' and row['ANSWER'] == '0':
                count_M_10 += 1
            elif row['PREDICTION'] == '1' and row['ANSWER'] == '1':
                count_M_11 += 1
        elif patient_dic[patient_id][1] == 'F':
            count_F += 1
            if row['PREDICTION'] == '0' and row['ANSWER'] == '0':
                count_F_00 += 1
            elif row['PREDICTION'] == '0' and row['ANSWER'] == '1':
                count_F_01 += 1
            elif row['PREDICTION'] == '1' and row['ANSWER'] == '0':
                count_F_10 += 1
            elif row['PREDICTION'] == '1' and row['ANSWER'] == '1':
                count_F_11 += 1
        
        if patient_dic[patient_id][2][:5] == 'WHITE':
            count_white += 1
            if row['PREDICTION'] == '0' and row['ANSWER'] == '0':
                count_white_00 += 1
            elif row['PREDICTION'] == '0' and row['ANSWER'] == '1':
                count_white_01 += 1
            elif row['PREDICTION'] == '1' and row['ANSWER'] == '0':
                count_white_10 += 1
            elif row['PREDICTION'] == '1' and row['ANSWER'] == '1':
                count_white_11 += 1
        elif patient_dic[patient_id][2][:5] == 'BLACK':
            count_black += 1
            if row['PREDICTION'] == '0' and row['ANSWER'] == '0':
                count_black_00 += 1
            elif row['PREDICTION'] == '0' and row['ANSWER'] == '1':
                count_black_01 += 1
            elif row['PREDICTION'] == '1' and row['ANSWER'] == '0':
                count_black_10 += 1
            elif row['PREDICTION'] == '1' and row['ANSWER'] == '1':
                count_black_11 += 1
        elif patient_dic[patient_id][2][:5] == 'ASIAN':
            count_asian += 1
            if row['PREDICTION'] == '0' and row['ANSWER'] == '0':
                count_asian_00 += 1
            elif row['PREDICTION'] == '0' and row['ANSWER'] == '1':
                count_asian_01 += 1
            elif row['PREDICTION'] == '1' and row['ANSWER'] == '0':
                count_asian_10 += 1
            elif row['PREDICTION'] == '1' and row['ANSWER'] == '1':
                count_asian_11 += 1

    print('P(1|1, M):', count_M_11 / (count_M_11 + count_M_01))
    print('P(1|1, F):', count_F_11 / (count_F_11 + count_F_01))
    print('P(0|0, M):', count_M_00 / (count_M_00 + count_M_10))
    print('P(0|0, F):', count_F_00 / (count_F_00 + count_F_10))

        
    
    print('P(1|1, WHITE):', count_white_11 / (count_white_11 + count_white_01))
    print('P(1|1, BLACK):', count_black_11 / (count_black_11 + count_black_01))
    print('P(1|1, ASIAN):', count_asian_11 / (count_asian_11 + count_asian_01))
    print('P(0|0, WHITE):', count_white_00 / (count_white_00 + count_white_10))
    print('P(0|0, BLACK):', count_black_00 / (count_black_00 + count_black_10))
    print('P(0|0, ASIAN):', count_asian_00 / (count_asian_00 + count_asian_10))


    print('P(1|M):', (count_M_10 + count_M_11) / count_M)
    print('P(1|F):', (count_F_10 + count_F_11) / count_F)

    print('P(1|WHITE):', (count_white_10 + count_white_11) / count_white)
    print('P(1|BLACK):', (count_black_10 + count_black_11) / count_black)
    print('P(1|ASIAN):', (count_asian_10 + count_asian_11) / count_asian)