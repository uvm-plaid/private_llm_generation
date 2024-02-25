#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:3
#SBATCH --cpus-per-task=1
#SBATCH --time=01-23:59:59
#SBATCH --job-name=DP_Dialog
#SBATCH --output=%x_%j.out
# SBATCH --mail-user=kngongiv@uvm.edu
#SBATCH --mail-type=ALL 
#SBATCH --mem=64G

# Request email to be sent at begin and end, and if fails
# change to the directory where you submitted this script
cd ${SLURM_SUBMIT_DIR}
# Executable section: echoing some Slurm data
echo "Running host:    ${SLURMD_NODENAME}"
echo "Assigned nodes:  ${SLURM_JOB_NODELIST}"
echo "Job ID:          ${SLURM_JOBID}"

# Stage 1 - Decoding-guided Distillation    
# cd stage1/summary/ 
# python3 main_input.py --device_id=0 --domain='news'   # 1_GenerateInitialDataset , Runtime 15:25:55 
# python3 main_input.py --device_id=0 --domain='bio'   # 1_GenerateInitialDataset , Runtime 15:25:55 
# python3 main_input.py --device_id=0 --domain='dialog'   # 1_GenerateInitialDataset , Runtime 15:25:55 

# python main_output.py --device_id=0 --con_domain='news' --con_model_name='gpt2-xl' --shard_start=0 --shard_size=50000 # 1_GenerateFullDatasetWithContextualContraints #Runtime 48hrs => 15% 7331 images generated
# python main_output.py --device_id=0 --con_domain='bio' --con_model_name='BioGPT-large' --shard_start=0 --shard_size=50000 # 1_GenerateFullDatasetWithContextualContraints #Runtime 48hrs => 15% 7331 images generated
# python main_output.py --device_id=0 --con_domain='dialog' --con_model_name='DialoGPT-large' --shard_start=0 --shard_size=50000 # 1_GenerateFullDatasetWithContextualContraints #Runtime 48hrs => 15% 7331 images generated

# cd stage1/train/data/
# python process.py --device_id=0  # 2_CombineAndSplitDatasets , Runtime 00:02:16

# cd stage1/train/
# CUDA_VISIBLE_DEVICES="0" accelerate launch --multi_gpu main.py --save_ckpt  --num_epochs=30 # 1_TrainStudentLM

# Train DP Model 
# cd stage_dp
# python dp_summary.py   # 2_DP_Model_Train

# Stage 2 - Self-Distillation
# Generate contextual constraints
# cd stage2/generate_and_filter/
# python3 main_input.py --device_id=0 --domain='news'
# python3 main_input.py --device_id=0 --domain='bio'   # 3_GenerateInitialDataset

# # # Generate pairs out of constraints
# cd stage2/generate_and_filter/
# # python main_output.py --device_id=0 --con_domain='news' --con_model_name='gpt2-xl' --shard_start=0 --shard_size=20000 # GeneratePairDataset #Runtime 48hrs => 15% 7331 images generated
# # python main_output.py --device_id=0 --con_domain='bio' --con_model_name='BioGPT-large' --shard_start=0 --shard_size=50000 # 3_GenerateFullDatasetWithContextualContraints 
# python main_output.py --device_id=0 --con_domain='bio' --con_model_name='vibrant-cherry-10' --shard_start=0 --shard_size=50000 # 3_GenerateFullDatasetWithContextualContraints 
# python main_output.py --device_id=0 --con_domain='bio' --con_model_name='sage-glade-19' --shard_start=0 --shard_size=50000 # 3_GenerateFullDatasetWithContextualContraints 
##### Change --con_model_name here, maybe change it to the path

# # Aggregate generated dataset 
# cd stage2/train/data/
# python process.py --device_id=1  # 3_CombineAndSplitDatasets

# # Train task model
# cd stage2/train/
# # CUDA_VISIBLE_DEVICES="0" accelerate launch --multi_gpu main.py --save_ckpt
# # CUDA_VISIBLE_DEVICES="0,1,2" accelerate launch --multi_gpu main.py --save_ckpt  # 3_Train_Final_Model
# CUDA_VISIBLE_DEVICES="0,1,2" accelerate launch --multi_gpu main.py  --model_name='sage-glade-19' --save_ckpt  # 3_Train_Final_Model

# Run with CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --multi_gpu main.py --save_ckpt

# Evaluate the Model on a test set
# cd stage2/train/
# python evaluate_model_pubmed.py  #4_EvaluationStageDPModel #4_EvaluationFinalModel

# Evaluate with GPT-4
# cd evaluation/gpt4/
# python gpt4_evaluation.py

# Evaluate with AlpacaEval
# cd evaluation/alpaca_eval/
# alpaca_eval evaluate --model_outputs='impossible_distil.json' --reference_outputs='dpsgd.json'

# alpaca_eval --model_outputs='impossible_distil.json' --reference_outputs='dpsgd.json'

# alpaca_eval --model_outputs='impossible_distil.json' 
# # alpaca_eval --model_outputs='dpsgd.json' 
# # alpaca_eval --model_outputs='dpsgd.json' --reference_outputs='impossible_distil.json'

###################
# cd dp
# python dp_summary.py #DP_Dialog
