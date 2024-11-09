# Results Reproduction
In this file, we will show how to reproduce the results in the paper.

## Dataset Detail
The process of making the dataset is as follows: firstly, the proportion of all the data is counted, and then the appropriate amount of data is selected in equal proportions as the temporary training set, validation set and test set with ratio of 5:1:4. After that, the training set is downsampled so that the amount of training data for each label is the same.

The random_seed is used to form five different data sets to realize the effect of 5 runs, corresponding to random_index 0 to 4.

And for the case of specifying a sample size of 500 test sets, the operation is the same as above, except that it is scaled down in equal proportions, corresponding to random_index 6.

The detailed data volume is shown in the table below:

- Length-of-Stay prediction, random_index 0 to 4

| label | 1 | 2 | 3 |
|-------|---|---|---|
| train | 2980 | 2980 | 2980 |
| val | 1200 | 596 | 425 |
| test | 2400 | 1192 | 851 |


- Mortality prediction, random_index 0 to 4

| label | 0 | 1 |
|-------|---|---|
| train | 2100 | 2100 |
| val | 2273 | 300 |
| test | 4546 | 600 |
| all | 22731 | 3000 |

- Readmission prediction, random_index 0 to 4

| label | 0 | 1 |
|-------|---|---|
| train | 277 | 277 |
| val | 500 | 40 |
| test | 1000 | 79 |
| all | 5000 | 396 |

- Length-of-Stay prediction, random_index 6

| label | 1 | 2 | 3 |
|-------|---|---|---|
| train | 335 | 335 | 335 |
| val | 135 | 67 | 48 |
| test | 270 | 134 | 96 |

- Mortality prediction, random_index 6

| label | 0 | 1 |
|-------|---|---|
| train | 204 | 204 |
| val | 221 | 29 |
| test | 442 | 58 |

- Readmission prediction, random_index 6

| label | 0 | 1 |
|-------|---|---|
| train | 128 | 128 |
| val | 232 | 18 |
| test | 463 | 37 |


## Table 1
In this table, the case of LLMs with the results of five runs of traditional ML models is represented.

For traditional ML models(take mortality prediction as an example):
```
python tradition.py \
	--task mortality_pred \
	--dataset mimic3\
	--random_index 0

python tradition.py \
	--task mortality_pred \
	--dataset mimic3\
	--random_index 1

python tradition.py \
	--task mortality_pred \
	--dataset mimic3\
	--random_index 2

python tradition.py \
	--task mortality_pred \
	--dataset mimic3\
	--random_index 3

python tradition.py \
	--task mortality_pred \
	--dataset mimic3\
	--random_index 4
```

For LLMs(take LLama3-Instruct in mortality prediction as an example):

```
python test_withprob.py \
	--base_model meta-llama/Meta-Llama-3-8B-Instruct \ 
	--dataset mimic3 \
	--task mortality_pred \
	--mode ORI \
	--random_index 0

python test_withprob.py \
	--base_model meta-llama/Meta-Llama-3-8B-Instruct \ 
	--dataset mimic3 \
	--task mortality_pred \
	--mode ORI \
	--random_index 1

python test_withprob.py \
	--base_model meta-llama/Meta-Llama-3-8B-Instruct \ 
	--dataset mimic3 \
	--task mortality_pred \
	--mode ORI \
	--random_index 2

python test_withprob.py \
	--base_model meta-llama/Meta-Llama-3-8B-Instruct \ 
	--dataset mimic3 \
	--task mortality_pred \
	--mode ORI \
	--random_index 3

python test_withprob.py \
	--base_model meta-llama/Meta-Llama-3-8B-Instruct \ 
	--dataset mimic3 \
	--task mortality_pred \
	--mode ORI \
	--random_index 4
```

The results will be saved as `results/{task}/{dataset}/{task}_result_data_{model_name}_{random_index}`

Use the `calculate.py` to calculate the F1 results and AUROC results. Then calculate the ranges of performance with 95% Confidence Interval to get the results in table 1.

## Table 2
In table 2, due to time constraints, instead of testing the results of 5 runs as in table1, only the results of 1 run (`random_index = 0`) are tested.

For traditional ML models(take mortality prediction as an example):
```
python tradition.py \
	--task mortality_pred \
	--dataset mimic3\
	--random_index 0
```

For LLMs(take LLama3-Instruct in mortality prediction as an example):

```
python test.py \
	--base_model meta-llama/Meta-Llama-3-8B-Instruct \ 
	--dataset mimic3 \
	--task mortality_pred \
	--mode ORI \
	--random_index 0
```

The results will be saved as `results/{task}/{dataset}/{task}_result_data_{model_name}_0`.

Then use the `calculate.py` to calculate the F1 results and AUROC results.

## Table 3
Unlike table1, table2, in table3 the test time is much longer due to some Prompt Engineering, so the test set size is controlled to be 500 samples with the same proportion as the real proportion. That is, specify `randon_index=6` when running the code.

For traditional ML models(take mortality prediction as an example):
```
python tradition.py \
	--task mortality_pred \
	--dataset mimic3\
	--random_index 6
```

For LLMs(take LLama3-Instruct in mortality prediction using ICL method as an example):

```
python test.py \
	--base_model meta-llama/Meta-Llama-3-8B-Instruct \ 
	--dataset mimic3 \
	--task mortality_pred \
	--mode ICL \
	--random_index 6
```

The results will be saved as `results/{task}/{dataset}/{task}_result_data_{model_name}_6`.

## Table 4
The same as table 1 but change the dataset to MIMIC-IV.

## Table 5, 6, 7, 8
In these tables, the results of training traditional ML Models with different proportions of training sets are represented.


Take the script of mortality prediction, on the MIMIC-III dataset, trained with 40% of the training set, as an example:

```
python tradition.py \
	--task mortality_pred \
	--dataset mimic3\
	--random_index 0\
    --ratio 0.4

python tradition.py \
	--task mortality_pred \
	--dataset mimic3\
	--random_index 1\
    --ratio 0.4

python tradition.py \
	--task mortality_pred \
	--dataset mimic3\
	--random_index 2\
    --ratio 0.4

python tradition.py \
	--task mortality_pred \
	--dataset mimic3\
	--random_index 3\
    --ratio 0.4

python tradition.py \
	--task mortality_pred \
	--dataset mimic3\
	--random_index 4\
    --ratio 0.4
```

The results will be saved as `results/{task}/{dataset}/{task}_result_data_{model_name}_{random_index}_{ratio}`

Use the `calculate.py` to calculate the F1 results and AUROC results. Then calculate the ranges of performance with 95% Confidence Interval to get the results in table 1.

## Figure 3
In this figure, the results of using different temperature to test LLMs are represented.

Take LLama3-Instruct, MIMIC-III dataset, mortality prediction task as an example:

```
python test.py \
	--base_model meta-llama/Meta-Llama-3-8B-Instruct \ 
	--dataset mimic3 \
	--task mortality_pred \
	--mode ORI \
	--random_index 0 \
	--temperature 0.2

python test.py \
	--base_model meta-llama/Meta-Llama-3-8B-Instruct \ 
	--dataset mimic3 \
	--task mortality_pred \
	--mode ORI \
	--random_index 0 \
	--temperature 0.4

python test.py \
	--base_model meta-llama/Meta-Llama-3-8B-Instruct \ 
	--dataset mimic3 \
	--task mortality_pred \
	--mode ORI \
	--random_index 0 \
	--temperature 0.6

python test.py \
	--base_model meta-llama/Meta-Llama-3-8B-Instruct \ 
	--dataset mimic3 \
	--task mortality_pred \
	--mode ORI \
	--random_index 0 \
	--temperature 0.8

python test.py \
	--base_model meta-llama/Meta-Llama-3-8B-Instruct \ 
	--dataset mimic3 \
	--task mortality_pred \
	--mode ORI \
	--random_index 0 \
	--temperature 1
```

The results will be saved as `results/{task}/{dataset}/{task}_result_data_{model_name}_0_{temperature}.csv`

## Figure 4
This figure shows the results of fine-tuning the LLMs. The division of the training, validation and test sets for fine-tuning is the same as in the previous case of setting `random_index=6` to facilitate training and comparison with previous results

We use [LLama Factory](https://github.com/hiyouga/LLaMA-Factory) to fine-tuning the models. 

The structure of training data is like this:

{'instruction': "Given the patient information, predict the number of weeks of stay in hospital.\nAnswer 1 if no more than one week,\nAnswer 2 if more than one week but not more than two weeks,\nAnswer 3 if more than two weeks.\nAnswer with only the number", 'input': input, 'output': output}

{'instruction': 'Given the patient information, predict the mortality of the patient.\nAnswer 1 if the patient will die, answer 0 otherwise.\nAnswer with only the number', 'input': input, 'output': output}

{'instruction': 'Given the patient information, predict the readmission of the patient.\nAnswer 1 if the patient will be readmitted to the hospital within two weeks, answer 0 otherwise.\nAnswer with only the number', 'input': input, 'output': output}