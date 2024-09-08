from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import torch
from tqdm.auto import tqdm
import numpy as np
from peft import PeftModel, PeftConfig
import transformers
from torch.nn import functional as F
base_model_name = "google/gemma-2-9b-it"

model_name = base_model_name.split('/')[1]

device = "cuda:0"

dataset = "mimic3"

random_index = 0



model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    device_map='auto',
    torch_dtype=torch.float16,
    trust_remote_code=True
)


count = 0
right = 0
the_bool = True
if base_model_name == "chaoyi-wu/MedLLaMA_13B":
    tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/MedLLaMA_13B', trust_remote_code=True)
else:
    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

# lora_model_name = '/home/peter/files/try/LLaMA-Factory/saves/llama3_length_31'
# lora_model = PeftModel.from_pretrained(model, lora_model_name)
# model = lora_model.merge_and_unload()

numbers = ['1', '2', '3']
number_token_ids = tokenizer.convert_tokens_to_ids(numbers)


with open(f'/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/test_index_{random_index}.npy', 'rb') as f:
    test_index = np.load(f)

test_index = test_index.tolist()

with open(f'/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/length_pred_result_data_{model_name}_{random_index}_ICL.csv', 'w') as file:
    filenames = ['SUBJECT_ID', 'ANSWER', 'PREDICTION', 'PROB']
    writer = csv.DictWriter(file, fieldnames=filenames)
    writer.writeheader()

    with open(f'/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/length_pred_data_ICL.csv', 'r') as f:
        total_rows = sum(1 for line in f) - 1
    with open(f'/home/peter/files/PyHealth/nlp_task/length_pred/{dataset}/length_pred_data_ICL.csv', 'r') as f:
        csvreader = csv.DictReader(f)
        for row in tqdm(csvreader, total=total_rows, desc="Processing"):
            if row["VISIT_ID"] not in test_index:
                continue
            # instruction = 'Given the patient information, predict the number of weeks of stay in hospital.\nAnswer 1 if no more than one week,\nAnswer 2 if more than one week but not more than two weeks,\nAnswer 3 if more than two weeks.\nAnswer with only the number'
            prompt = row['QUESTION']
            # prompt = instruction + '\n' + prompt[:-215] + '\nAnswer: '
            # prompt = 'user\n\n' + prompt + 'assistant\n\n'
            gt = row['ANSWER']
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
            # outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1, output_scores=True, return_dict_in_generate=True)
            # last_token_scores = outputs.scores[-1][0]  # shape: [vocab_size]

            # top_k = 5
            # top_k_scores, top_k_indices = torch.topk(last_token_scores, k=top_k)

            # for i in range(top_k):
            #     token = tokenizer.decode([top_k_indices[i]])
            #     score = top_k_scores[i].item()
            #     print(f"Token: {token}, Score: {score}")

            # print("Logits for specific number tokens:")
            # for number, token_id in zip(numbers, number_token_ids):
            #     score = last_token_scores[token_id].item()
            #     print(f"Number: {number}, Token ID: {token_id}, Score: {score}")
            # breakpoint()
            # response = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
            # original_response = response
            with torch.no_grad():
                outputs = model(**inputs)

            # 获取最后一个token的logits
            last_token_logits = outputs.logits[0, -1]

            number_logits = last_token_logits[number_token_ids]
            probabilities = F.softmax(number_logits, dim=0)
            most_likely_token_id = torch.argmax(last_token_logits).item()
            predicted_token = tokenizer.decode([most_likely_token_id])
            for number, token_id, prob in zip(numbers, number_token_ids, probabilities):
                logit = last_token_logits[token_id].item()
                probability = prob.item()
                # print(f"Number: {number}, Logit: {logit:.4f}, Probability: {probability:.4f}")
            response = predicted_token
            # if response not in ['1', '2', '3']:
            #     continue
            count += 1
            try:
                response = int(response)
            except:
                continue
            if int(response) == int(gt):
                right += 1
            writer.writerow({'SUBJECT_ID': row['SUBJECT_ID'], 'ANSWER': row['ANSWER'], 'PREDICTION': response, 'PROB': probability})
            
    print(f"Accuracy: {right/count}")



    # no sample: Accuracy 0.12660166190015087
    # do sample: Accuracy 0.11426126511586011