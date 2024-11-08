# Results Reproduction
In this file, we will show how to reproduce the results in the paper.

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

`results/{task}/{dataset}/{task}_result_data_{model_name}_{random_index}_{ratio}`

Use the `calculate.py` to calculate the F1 results and AUROC results. Then calculate the ranges of performance with 95% Confidence Interval to get the results in table 1.

