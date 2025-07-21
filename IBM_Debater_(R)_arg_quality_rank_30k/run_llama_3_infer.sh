#!/bin/bash
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=3
#SBATCH --array=1-5
#SBATCH --gres=gpu:4
#SBATCH --mem=100G
#SBATCH --job-name=test_llama_infer
#SBATCH --output=class_llama_infer_%a.out
#SBATCH --error=error_llama_infer_%a.out
#SBATCH --account=def-mageed
#SBATCH --mail-user=rrs99@cs.ubc.ca
#SBATCH --mail-type=ALL

module load StdEnv/2023 python/3.10 scipy-stack/2023a cuda arrow/17.0.0
module load cudnn
source /scratch/rrs99/venv/venv_llama/bin/activate

export DIR_INDEX=$SLURM_ARRAY_TASK_ID

python3 infer_llama.py --dir_index=$DIR_INDEX