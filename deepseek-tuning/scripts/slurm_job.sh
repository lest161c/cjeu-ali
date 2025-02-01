#!/bin/bash
#SBATCH --job-name=deepseek-vllm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:4
#SBATCH --time=48:00:00
#SBATCH --output=slurm-%j.out

# Run fine-tuning with axolotl
docker run --rm --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/models:/workspace/models \
  -v $(pwd)/finetuned_model:/workspace/finetuned_model \
  deepseek-vllm \
  axolotl train /workspace/axolotl_config.yaml

# Run vLLM inference on test data
docker run --rm --gpus all \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/finetuned_model:/workspace/finetuned_model \
  deepseek-vllm \
  python vllm_test.py --model /workspace/finetuned_model --data /workspace/data/test.jsonl