from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import torch
from tqdm.auto import tqdm
import numpy as np
from peft import PeftModel, PeftConfig
import transformers
from torch.nn import functional as F
# base_model_name = "m42-health/Llama3-Med42-8B"

# dataset = "mimic4"
# random_index = 0

def main(base_model_name, dataset, random_index):

    model_name = base_model_name.split('/')[1]

    device = "cuda:0"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map='auto',
        torch_dtype=torch.float16,
        trust_remote_code=True
    )


    count = 0
    right = 0
    if base_model_name == "chaoyi-wu/MedLLaMA_13B":
        tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/MedLLaMA_13B', trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)

    # lora_model_name = '/home/peter/files/try/LLaMA-Factory/saves/llama3_length_31'
    # lora_model = PeftModel.from_pretrained(model, lora_model_name)
    # model = lora_model.merge_and_unload()

    numbers = ['0', '1']
    number_token_ids = tokenizer.convert_tokens_to_ids(numbers)

    with open(f'/home/peter/files/PyHealth/nlp_task/mortality_pred/{dataset}/test_index_{random_index}.npy', 'rb') as f:
        test_index = np.load(f)

    test_index = test_index.tolist()

    with open(f'/home/peter/files/PyHealth/nlp_task/mortality_pred/{dataset}/mortality_pred_result_data_{model_name}_{random_index}.csv', 'w') as file:
        filenames = ['SUBJECT_ID', 'ANSWER', 'PREDICTION', 'PROB']
        writer = csv.DictWriter(file, fieldnames=filenames)
        writer.writeheader()

        with open(f'/home/peter/files/PyHealth/nlp_task/mortality_pred/{dataset}/mortality_pred_data.csv', 'r') as f:
            total_rows = sum(1 for line in f) - 1
        with open(f'/home/peter/files/PyHealth/nlp_task/mortality_pred/{dataset}/mortality_pred_data.csv', 'r') as f:
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
                # outputs = model.generate(**inputs, max_new_tokens=1, do_sample=False, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1)
                # response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                # original_response = response
                # response = response[-1]
                with torch.no_grad():
                    outputs = model(**inputs)

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

                count += 1
                try:
                    response = int(response)
                except:
                    continue
                if int(response) == int(gt):
                    right += 1
                writer.writerow({'SUBJECT_ID': row['SUBJECT_ID'], 'ANSWER': row['ANSWER'], 'PREDICTION': response, 'PROB': probability})
                
        print(f"Accuracy: {right/count}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="运行模型与指定参数")
    parser.add_argument("--base_model_name", type=str, required=True, help="基础模型名称")
    parser.add_argument("--dataset", type=str, required=True, help="数据集名称")
    parser.add_argument("--random_index", type=int, required=True, help="随机索引")
    
    args = parser.parse_args()
    
    main(args.base_model_name, args.dataset, args.random_index)