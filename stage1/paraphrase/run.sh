#!/bin/bash

#SBATCH --partition=dggpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --time=01-23:59:59
#SBATCH --job-name=1_GenParaphraseFULLDataset
#SBATCH --output=%x_%j.out
#SBATCH --mail-user=kngongiv@uvm.edu
#SBATCH --mail-type=ALL 

# SBATCH --mem=14G

# Request email to be sent at begin and end, and if fails
# change to the directory where you submitted this script
cd ${SLURM_SUBMIT_DIR}
# Executable section: echoing some Slurm data
echo "Running host:    ${SLURMD_NODENAME}"
echo "Assigned nodes:  ${SLURM_JOB_NODELIST}"
echo "Job ID:          ${SLURM_JOBID}"

#  Stage 1 - Decoding-guided Distillation    
# cd stage1/summary/ 
# python3 main_input.py --device_id=0 --domain='news' --number=10  # 1_GenParaphraseDataset , Runtime 15:25:55 

python main_output.py --device_id=0 --con_domain='news' --con_model_name='gpt2-xl' --shard_start=0 --shard_size=90020 # 1_GenParaphraseFULLDataset #Runtime 48hrs => 15% 7331 images generated

# cd stage1/train/data/
# python process.py --device_id=0  # CombineDatasets

