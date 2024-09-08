import subprocess
import itertools

# 定义参数列表
base_models = ["lmsys/vicuna-13b-v1.5", "chaoyi-wu/MedLLaMA_13B"]

datasets = ["mimic3", "mimic4"]
random_indices = [0, 1, 2, 3, 4]

# 指定 GPU
gpu_id = 0,2  # 根据需要修改 GPU ID

# 主脚本的文件名
main_script = "test2.py"

cleanup_command = f"CUDA_VISIBLE_DEVICES=0,2 python -c \"import torch; torch.cuda.empty_cache()\""

# 遍历所有参数组合
for base_model, dataset, random_index in itertools.product(base_models, datasets, random_indices):

    # 清理 GPU
    subprocess.run(cleanup_command, shell=True, check=True)
    
    # 构建命令
    command = f"CUDA_VISIBLE_DEVICES=0,2 python {main_script} " \
              f"--base_model_name '{base_model}' " \
              f"--dataset '{dataset}' " \
              f"--random_index {random_index}"
    
    print(f"执行命令: {command}")
    
    # 执行命令
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"执行失败: {e}")

print("所有任务完成")
