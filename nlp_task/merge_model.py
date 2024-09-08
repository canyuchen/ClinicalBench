from transformers import AutoModelForCausalLM
from peft import PeftModel, PeftConfig

# 加载原始模型
base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")

# 加载LoRA配置
peft_config = PeftConfig.from_pretrained("/home/peter/files/try/LLaMA-Factory/saves/llama3-8b/lora/llama_instruct_length_full/checkpoint-6000")

# 加载LoRA模型
lora_model = PeftModel.from_pretrained(base_model, "/home/peter/files/try/LLaMA-Factory/saves/llama3-8b/lora/llama_instruct_length_full/checkpoint-6000")

# 合并权重
merged_model = lora_model.merge_and_unload()

# 保存合并后的模型
merged_model.save_pretrained("/home/peter/files/PyHealth/nlp_task/length_pred/llama-Instruct-FT")