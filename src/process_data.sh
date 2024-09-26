#!/bin/bash

MIMIC3PATH=""
MIMIC4PATH=""


echo "Convert EHR data into clinical codes..."
python get_data.py \
    --mimic3_path ${MIMIC3PATH} \
    --mimic4_path ${MIMIC4PATH}
if [ $? -ne 0 ]; then
    echo "Error running get_data.py"
fi


# mimic3, length
cd data/length_pred/mimic3

echo "Preprocess prompt for length prediction, mimic3..."
python convert.py \
    --dataset_path ${MIMIC3PATH} \
    --ICL 0
if [ $? -ne 0 ]; then
    echo "Error running convert.py for length prediction, mimic3"
fi

echo "Preprocess prompt for length prediction, mimic3, ICL..."
python convert.py \
    --dataset_path ${MIMIC3PATH} \
    --ICL 1
if [ $? -ne 0 ]; then
    echo "Error running convert.py for length prediction, mimic3, ICL"
fi

# mimic4, length
cd ../mimic4

echo "Preprocess prompt for length prediction, mimic4..."
python convert.py\
    --dataset_path ${MIMIC4PATH} \
    --ICL 0
if [ $? -ne 0 ]; then
    echo "Error running convert.py for length prediction, mimic4"
fi

echo "Preprocess prompt for length prediction, mimic4, ICL..."
python convert.py\
    --dataset_path ${MIMIC4PATH} \
    --ICL 1
if [ $? -ne 0 ]; then
    echo "Error running convert.py for length prediction, mimic4, ICL"
fi


cd ../../../

# mimic3, mortality
cd data/mortality_pred/mimic3
echo "Preprocess prompt for mortality prediction, mimic3..."
python convert.py\
    --dataset_path ${MIMIC3PATH} \
    --ICL 0
if [ $? -ne 0 ]; then
    echo "Error running convert.py for mortality prediction, mimic3"
fi

echo "Preprocess prompt for mortality prediction, mimic3, ICL..."
python convert.py\
    --dataset_path ${MIMIC3PATH} \
    --ICL 1
if [ $? -ne 0 ]; then
    echo "Error running convert.py for mortality prediction, mimic3, ICL"
fi

# mimic4, mortality
cd ../mimic4
echo "Preprocess prompt for mortality prediction, mimic4..."
python convert.py\
    --dataset_path ${MIMIC4PATH} \
    --ICL 0
if [ $? -ne 0 ]; then
    echo "Error running convert.py for mortality prediction, mimic4"
fi

echo "Preprocess prompt for mortality prediction, mimic4, ICL..."
python convert.py\
    --dataset_path ${MIMIC4PATH} \
    --ICL 1
if [ $? -ne 0 ]; then
    echo "Error running convert.py for mortality prediction, mimic4, ICL"
fi

cd ../../../

# mimic3, readmission
cd data/readmission_pred/mimic3
echo "Preprocess prompt for readmission prediction, mimic3..."
python convert.py\
    --dataset_path ${MIMIC3PATH} \
    --ICL 0
if [ $? -ne 0 ]; then
    echo "Error running convert.py for readmission prediction, mimic3"
fi

echo "Preprocess prompt for readmission prediction, mimic3, ICL..."
python convert.py\
    --dataset_path ${MIMIC3PATH} \
    --ICL 1
if [ $? -ne 0 ]; then
    echo "Error running convert.py for readmission prediction, mimic3, ICL"
fi


# mimic4, readmission
cd ../mimic4
echo "Preprocess prompt for readmission prediction, mimic4..."
python convert.py\
    --dataset_path ${MIMIC4PATH} \
    --ICL 0
if [ $? -ne 0 ]; then
    echo "Error running convert.py for readmission prediction, mimic4"
fi

echo "Preprocess prompt for readmission prediction, mimic4, ICL..."
python convert.py\
    --dataset_path ${MIMIC4PATH} \
    --ICL 1
if [ $? -ne 0 ]; then
    echo "Error running convert.py for readmission prediction, mimic4, ICL"
fi
cd ../../../


echo "Done."

