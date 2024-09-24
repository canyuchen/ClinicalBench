from transformers import AutoModelForCausalLM, AutoTokenizer
import csv
import torch
from tqdm.auto import tqdm
import numpy as np
from peft import PeftModel, PeftConfig
import transformers
from torch.nn import functional as F
import argparse
from sklearn.metrics import f1_score
from transformers.cache_utils import Cache



def main(args):
    base_model_name = args.base_model
    model_name = base_model_name.split('/')[-1]
    dataset = args.dataset
    task = args.task
    random_index = args.random_index
    mode = args.mode
    temperature = args.temperature
    cache_dir = args.cache_dir
    lora_path = args.lora_path
    

    mode_str = ""
    data_str = ""
    if mode == "ICL":
        mode_str = "_ICL"
        data_str = "_ICL"
        random_index = 6
    if mode == "COT":
        mode_str = "_COT"
        random_index = 6
    if mode == "RP":
        mode_str = "_RP"
        random_index = 6
    if mode == "SR":
        mode_str = "_SR"
        random_index = 6
    if mode == "LORA":
        mode_str = "_LORA"
    temp_str = ""
    if temperature:
        temp_str = "_temp_" + str(temperature)
        
    do_sample = False
    if temperature is not None:
        do_sample = True

    device = "cuda:0"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map='auto',
        torch_dtype=torch.float16,
        trust_remote_code=True,
        cache_dir=cache_dir
    )



    if base_model_name == "chaoyi-wu/MedLLaMA_13B":
        tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/MedLLaMA_13B', trust_remote_code=True, cache_dir=cache_dir)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True, cache_dir=cache_dir)


    if mode == "LORA":
        lora_model = PeftModel.from_pretrained(model, lora_path)
        model = lora_model.merge_and_unload()


    with open(f'data/{task}/{dataset}/test_index_{random_index}.npy', 'rb') as f:
        test_index = np.load(f)

    test_index = test_index.tolist()

    preds = []
    answers = []
    
    number_list = ['1', '2', '3'] if task == "length_pred" else ['0', '1']
    number_token_ids = tokenizer.convert_tokens_to_ids(number_list)
    with open(f'results/{task}/{dataset}/{task}_result_data_{model_name}_{random_index}{mode_str}{temp_str}.csv', 'w') as file:
        filenames = ['SUBJECT_ID', 'ANSWER', 'PREDICTION', 'PROB']
        writer = csv.DictWriter(file, fieldnames=filenames)
        writer.writeheader()

        with open(f'data/{task}/{dataset}/{task}_data{data_str}.csv', 'r', newline='') as f:
            csv_reader = csv.reader(f)
            next(csv_reader)
            total_rows = sum(1 for row in csv_reader)
        with open(f'data/{task}/{dataset}/{task}_data{data_str}.csv', 'r') as f:
            csvreader = csv.DictReader(f)
            for row in tqdm(csvreader, total=total_rows, desc="Processing"):
                if row["VISIT_ID"] not in test_index:
                    continue
                prompt = row['QUESTION']
                # change the prompt based on the mode
                if mode == "COT":
                    if task == "length_pred":
                        cut_length = len("\nAnswer 1 if no more than one week,\nAnswer 2 if more than one week but not more than two weeks,\nAnswer 3 if more than two weeks.\nAnswer with only the number. Answer: ")
                        prompt = prompt[:-cut_length]
                        prompt = prompt + "\nPlease provide your concise reasoning steps for the prediction(no more than 3 steps), and finally answer 1 if the patient will stay no more than one week, answer 2 if more than one week but not more than two weeks, answer 3 if more than two weeks."
                    if task == "readmission_pred":
                        cut_length = len("\nAnswer 1 for yes, 0 for no. Answer with only the number.\nAnswer: ")
                        prompt = prompt[:-cut_length]
                        prompt = prompt + "\nPlease provide your concise reasoning steps for the prediction(no more than 3 steps), and finally answer 1 if the patient will be readmitted and 0 otherwise."
                    if task == "mortality_pred":
                        cut_length = len("\nAnswer 1 if yes, 0 if no. Answer with only the number.\nAnswer: ")
                        prompt = prompt[:-cut_length]
                        prompt = prompt + "\nPlease provide your concise reasoning steps for the prediction(no more than 3 steps), and finally answer 1 if the patient will die and 0 otherwise."
                if mode == "RP":
                    role = "Imagine that you are a doctor. Today, you're seeing a patient with the following profile:\n"
                    prompt = role + prompt
                if mode == "SR":
                    if task == "length_pred":
                        cut_length = len("\nAnswer with only the number. Answer: ")
                    elif task == "mortality_pred":
                        cut_length = len("Answer with only the number.\nAnswer: ")
                    elif task == "readmission_pred":
                        cut_length = len("Answer with only the number.\nAnswer: ")
                    prompt = prompt[:-cut_length]
                    self_reflection = "First answer with a number. Then conduct a concise reflection. Finally output your answer again with a number."
                    prompt = prompt + "\n" + self_reflection
                gt = row['ANSWER']
                inputs = tokenizer(prompt, return_tensors="pt")
                inputs = {name: tensor.to(device) for name, tensor in inputs.items()}
                with torch.no_grad():
                    outputs = model(**inputs)
                predicted_token_ids = outputs.logits[0, :].argmax(dim=-1)
                predicted_text = tokenizer.decode(predicted_token_ids)
                number_position = 1
                for i in range(len(predicted_text)+1):
                    if predicted_text[-i] in number_list:
                        number_position = i
                        break
                number_token_logits = outputs.logits[0, -number_position]
                number_logits = number_token_logits[number_token_ids]
                probabilities = F.softmax(number_logits, dim=0)
                most_likely_token_id = number_token_ids[torch.argmax(number_logits).item()]
                predicted_token = tokenizer.decode([most_likely_token_id])
                probability = float(probabilities[-1])
                response = predicted_token
                
                preds.append(int(response))
                answers.append(int(gt))
                
                
                writer.writerow({'SUBJECT_ID': row['SUBJECT_ID'], 'ANSWER': row['ANSWER'], 'PREDICTION': response, 'PROB': probability})
        
        if task == "length_pred":
            f1 = f1_score(answers, preds, average="macro")
        else:
            f1 = f1_score(answers, preds)
        print(len(answers))
        print(f"F1: {f1}")
            






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM Prediction Script")
    parser.add_argument("--base_model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct", help="Base model name")
    parser.add_argument("--lora_path", type=str, default=None, help="The path to lora model")
    parser.add_argument("--dataset", type=str, default="mimic3", help="Dataset name")
    parser.add_argument("--task", type=str, default="length_pred", help="Task name")
    parser.add_argument("--random_index", type=int, default=0, help="Random index")
    parser.add_argument("--mode", type=str, default="ORI", choices=["ORI", "ICL", "COT", "RP", "SR", "LORA"], help="Mode")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for sampling")
    parser.add_argument("--cache_dir", type=str, default="", help="Cache directory")

    args = parser.parse_args()
    main(args)