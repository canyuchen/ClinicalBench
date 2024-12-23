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



def main(args):
    base_model_name = args.base_model
    model_name = base_model_name.split('/')[-1]
    dataset = args.dataset
    task = args.task
    random_index = args.random_index
    mode = args.mode
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

    device = "cuda:0"

    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        device_map='auto',
        torch_dtype=torch.float16,
        trust_remote_code=True,
    )



    if base_model_name == "chaoyi-wu/MedLLaMA_13B":
        tokenizer = transformers.LlamaTokenizer.from_pretrained('chaoyi-wu/MedLLaMA_13B', trust_remote_code=True)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)


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
    with open(f'results/{task}/{dataset}/{task}_result_data_{model_name}_{random_index}{mode_str}.csv', 'w') as file:
        filenames = ['SUBJECT_ID', 'ANSWER', 'PREDICTION', 'ORIGINAL', 'PROB']
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
                predicted_text = tokenizer.decode(predicted_token_ids)[-1]
                original_response = predicted_text
                number_token_logits = outputs.logits[0, -1]
                number_logits = number_token_logits[number_token_ids]
                probabilities = F.softmax(number_logits, dim=0)
                # most_likely_token_id = number_token_ids[torch.argmax(number_logits).item()]
                # predicted_token_text = tokenizer.decode([most_likely_token_id])
                probability = float(probabilities[-1])
                response = predicted_text
                if response not in number_list:
                    if task == "length_pred":
                        if gt == "1":
                            response = "2"
                        else:
                            response = "1"
                    else:
                        if gt == "1":
                            response = "0"
                        else:
                            response = "1"
                
                
                preds.append(int(response))
                answers.append(int(gt))
                
                
                writer.writerow({'SUBJECT_ID': row['SUBJECT_ID'], 'ANSWER': row['ANSWER'], 'PREDICTION': response, 'ORIGINAL': original_response, 'PROB': probability})
        
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

    args = parser.parse_args()
    main(args)