#!/bin/bash

# SBATCH --partition=dggpu
#SBATCH --partition=bdgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=1
#SBATCH --time=01-23:59:59
#SBATCH --job-name=Hal_XSum_Vanilla
#SBATCH --output=out_eval/%x_%j.out
#SBATCH --mail-user=kngongiv@uvm.edu
#SBATCH --mail-type=ALL 
#SBATCH --mem=64G

# Request email to be sent at begin and end, and if fails
# change to the directory where you submitted this script
cd ${SLURM_SUBMIT_DIR}
# Executable section: echoing some Slurm data
echo "Running host:    ${SLURMD_NODENAME}"
echo "Assigned nodes:  ${SLURM_JOB_NODELIST}"
echo "Job ID:          ${SLURM_JOBID}"


### Hallucination
cd evaluation/
# python hallucination.py -d ./alpaca_eval/data/dp_results_xsum.json -o ./alpaca_eval/data/impossible_results_xsum.json -f ./hallucination_results/hallucinations_xsum.pdf -s ./hallucination_results/xsum_pii.json
# python hallucination.py -d ./alpaca_eval/data/dp_pubmeds.json -o ./alpaca_eval/data/impossible_pubmeds.json -f ./hallucination_results/hallucinations_pubmed.pdf -s ./hallucination_results/pubmed_pii.json
# python hallucination.py -d ./alpaca_eval/data/dp_mrpc.json -o ./alpaca_eval/data/impossible_mrpc.json -f ./hallucination_results/hallucinations_mrpc.pdf -s ./hallucination_results/mrpc_pii.json

# python hallucination_vanilla.py -d ./alpaca_eval/data/vanilla_mrpc.json -o ./alpaca_eval/data/impossible_mrpc.json -f ./hallucination_results/hallucination_vanilla_mrpc.pdf -s ./hallucination_results/mrpc_pii_vanilla.json
python hallucination_vanilla.py -d ./alpaca_eval/data/vanilla_xsum.json -o ./alpaca_eval/data/vanilla_xsum.json -f ./hallucination_results/hallucination_vanilla_xsum.pdf -s ./hallucination_results/xsum_pii_vanilla.json
