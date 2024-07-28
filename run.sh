#!/bin/bash

#SBATCH --partition=dggpu
# SBATCH --partition=bdgpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:5
#SBATCH --cpus-per-task=1
#SBATCH --time=01-23:59:59
#SBATCH --job-name=Stage_DP_XSUM_gen
# SBATCH --job-name=XSum_Model_Eps_8
#SBATCH --output=output/%x_%j.out
#SBATCH --mail-user=@uvm.edu
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

# python main_output.py --device_id=0 --con_domain='news' --con_model_name='gpt2-xl' --shard_start=0 --shard_size=50030 # 1_GenerateFullDatasetWithContextualContraints #Runtime 48hrs => 15% 7331 images generated
# python main_output.py --device_id=0 --con_domain='bio' --con_model_name='BioGPT-large' --shard_start=0 --shard_size=50000 # 1_GenerateFullDatasetWithContextualContraints #Runtime 48hrs => 15% 7331 images generated
# python main_output.py --device_id=0 --con_domain='dialog' --con_model_name='DialoGPT-large' --shard_start=0 --shard_size=50000 # 1_GenerateFullDatasetWithContextualContraints #Runtime 48hrs => 15% 7331 images generated

# cd stage1/train/data/
# python process.py --device_id=0  # 1_CombineAndSplitDatasets , Runtime 00:02:16

# cd stage1/train/
# cd ..
# CUDA_VISIBLE_DEVICES="0" accelerate launch --multi_gpu main.py --save_ckpt # --num_epochs=30 # 1_TrainStudentLM
# CUDA_VISIBLE_DEVICES="1" accelerate launch --multi_gpu main.py --save_ckpt --num_epochs=8 # 1_TrainStudentLM_8
# CUDA_VISIBLE_DEVICES="1" accelerate launch --main_process_port 29501 --multi_gpu main.py --save_ckpt --num_epochs=8
# CUDA_VISIBLE_DEVICES="1" accelerate launch --main_process_port 29501 --multi_gpu main.py --save_ckpt --num_epochs=5


# Train DP Model 
# cd stage_dp
# python dp_summary_xsum.py   # 2_StageDP_XSum_Train
# python dp_summary_samsum.py   # 2_DP_SamSum_Train
# python dp_summary_dialogsum.py # 2_DP_DialogSum_Train
# python dp_paraphrase_mrpc.py   # 2_Paraphrase_StageDP_MRPC

### Paraphrasing Tasks
# python dp_paraphrase_stagedp.py --project_name DP_Paraphrase_MRPC --dataset_name "mrpc" --run_name "mrpc_stagedp_epsilon_8" --target_epsilon 8 --student_model_dir "/users/k/n/kngongiv/Research/clean/private_llm_generation/stage1/train/gen_model/stage1_mrpc_model"
# python dp_paraphrase_stagedp.py --project_name DP_Summarization_MRPC --dataset_name "mrpc" --run_name "mrpc_stagedp_epsilon_8" --target_epsilon 8 --student_model_dir "/users/k/n/kngongiv/Research/clean/private_llm_generation/stage1/train/gen_model/stage1_mrpc_model"

### Summarization Tasks
# python dp_summary_stagedp.py --project_name DP_Summarization_XSum --dataset_name "xsum" --run_name "xsum_stagedp_epsilons_8" --target_epsilon 8 --epochs 2 --student_model_dir "/users/k/n/kngongiv/Research/clean/private_llm_generation/stage1/train/gen_model/stage1_xsum_model"
## python dp_summary_stagedp.py --project_name DP_Summarization_XSum --dataset_name "xsum" --run_name "xsum_stagedp_epsilons_8" --target_epsilon 8 --subset_size 40000 --student_model_dir "/users/k/n/kngongiv/Research/clean/private_llm_generation/stage1/train/gen_model/stage1_xsum_model"
# python dp_summary_stagedp.py --project_name DP_Summarization_PubMed --dataset_name "pubmed" --run_name "pubmed_stagedp_epsilons_8" --target_epsilon 8  --student_model_dir "/users/k/n/kngongiv/Research/clean/private_llm_generation/stage1/train/gen_model/stage1_pubmed_model"

# Stage 2 - Self-Distillation
# Generate contextual constraints
cd stage2/generate_and_filter/
# python3 main_input.py --device_id=0 --domain='news' --number=10
# python3 main_input.py --device_id=0 --domain='bio'   # 3_GenerateInitialDataset
# python3 main_input.py --device_id=0 --domain='dialog' --number=1  # 3_GenerateInitialDataset , Runtime 15:25:55 

# # Generate pairs out of constraints
# cd stage2/generate_and_filter/
# python main_output.py --device_id=0 --con_domain='news' --con_model_name='gpt2-xl' --shard_start=0 --shard_size=60000 # GeneratePairDataset #Runtime 48hrs => 15% 7331 images generated
# # python main_output.py --device_id=0 --con_domain='bio' --con_model_nasme='BioGPT-large' --shard_start=0 --shard_size=50000 # 3_GenerateFullDatasetWithContextualContraints 
# python main_output.py --device_id=0 --con_domain='bio' --con_model_name='vibrant-cherry-10' --shard_start=0 --shard_size=50000 # 3_GenerateFullDatasetWithContextualContraints 
# python main_output.py --device_id=0 --con_domain='bio' --con_model_name='sage-glade-19' --shard_start=0 --shard_size=50000 # 3_GenerateFullDatasetWithContextualContraints

# python main_output.py --device_id=0 --con_domain='news' --con_model_name='xsum_stage_dp' --shard_start=0 --shard_size=90020 # 3_GenerateFullDatasetWithContextualContraints
# python main_output.py --device_id=0 --con_domain='news' --con_model_name='xsum_dp' --shard_start=0 --shard_size=700015  # 3_GenerateFullDatasetWithContextualContraints  # NEXT
# python main_output.py --device_id=0 --con_domain='news' --con_model_name='xsum_stage_dp_full' --shard_start=0 --shard_size=100000  # 3_GenerateFullDatasetWithContextualContraints  # NEXT
# python main_output.py --device_id=0 --con_domain='news' --con_model_name='xsum_stage_dp_eps3' --shard_start=0 --shard_size=100015
python main_output.py --device_id=0 --con_domain='news' --con_model_name='xsum_stagedp_epsilons_8' --shard_start=0 --shard_size=100045


# python main_output.py --device_id=0 --con_domain='news' --con_model_name='mrpc_stage_dp_full' --shard_start=0 --shard_size=100020 # 3_GenerateFullDatasetWithContextualContraints  # NEXT
# python main_output.py --device_id=0 --con_domain='news' --con_model_name='mrpc_stage_dp_eps3' --shard_start=0 --shard_size=100050 # 3_GenerateFullDatasetWithContextualContraints  # NEXT

# python main_output.py --device_id=0 --con_domain='news' --con_model_name='stage1_xsum_model' --shard_start=0 --shard_size=100015  # 3_GenerateFullDatasetWithContextualContraints  # NEXT
# python main_output.py --device_id=0 --con_domain='news' --con_model_name='stage1_mrpc_model' --shard_start=0 --shard_size=100006 # 3_GenerateFullDatasetWithContextualContraints  # NEXT


###### 
# cd stage2/generate_and_filter/
# python main_output.py --device_id=0 --con_domain='dialog' --con_model_name='samsum_stage_dp_1' --shard_start=0 --shard_size=50015 # GeneratePairDataset #Runtime 48hrs => 15% 7331 samples generated # 3_SamsumStageDP1_PairDataset
# python main_output.py --device_id=0 --con_domain='dialog' --con_model_name='dialogsum_stage_dp_1' --shard_start=0 --shard_size=50015 # GeneratePairDataset #Runtime 48hrs => 15% 7331 samples generated # # 3_DialogStageDP1_PairDataset

# python main_output.py --device_id=0 --con_domain='dialog' --con_model_name='samsum_dp' --shard_start=0 --shard_size=50015 # GeneratePairDataset #Runtime 48hrs => 15% 7331 samples generated # 3_SamsumStageDP1_PairDataset
# python main_output.py --device_id=0 --con_domain='dialog' --con_model_name='dialogsum_dp' --shard_start=0 --shard_size=50015 # GeneratePairDataset #Runtime 48hrs => 15% 7331 samples generated # # 3_DialogStageDP1_PairDataset


##### Change --con_model_name here, maybe change it to the path

# # Aggregate generated dataset 
# cd stage2/train/data/
# python process.py --device_id=1 --model_name='samsum_stage_dp' #'samsum_dp' # 3_CombineAndSplitDatasets
# python process.py --device_id=1 --model_name='dialogsum_stage_dp' #'samsum_dp' # 3_CombineAndSplitDatasets
# python process.py --device_id=1 --model_name='samsum_dp'  # 3_CombineAndSplitDatasets
# python process.py --device_id=1 --model_name='dialogsum_dp'  # 3_CombineAndSplitDatasets

# python process.py --device_id=1 --model_name='xsum_dp'  # 3_CombineAndSplitDatasets
# python process.py --device_id=1 --model_name='xsum_stage_dp'  # 3_CombineAndSplitDatasets
# python process.py --device_id=1 --model_name='xsum_stage_dp_full'  # 3_CombineAndSplitDatasets
# python process.py --device_id=1 --model_name='mrpc_stage_dp_full'  # 3_CombineAndSplitDatasets
# python process.py --device_id=1 --model_name='xsum_stage_dp_eps3'  # 3_CombineAndSplitDatasets

# # Train task model
# cd stage2/train/
# # CUDA_VISIBLE_DEVICES="0" accelerate launch --multi_gpu main.py --save_ckpt
# # CUDA_VISIBLE_DEVICES="0,1,2" accelerate launch --multi_gpu main.py --save_ckpt  # 3_Train_Final_Model
# CUDA_VISIBLE_DEVICES="0,1,2" accelerate launch --multi_gpu main.py  --model_name='sage-glade-19' --save_ckpt  # 3_Train_Final_Model

# cd stage2/train/
# CUDA_VISIBLE_DEVICES="0,1,2" accelerate launch --multi_gpu main.py  --model_name='samsum_stage_dp' --save_ckpt --num_epochs=3 # 3_FinalSamsumStageDP
# CUDA_VISIBLE_DEVICES="0,1,2" accelerate launch --multi_gpu main.py  --model_name='dialogsum_stage_dp' --save_ckpt  # 3_FinalDialogStageDP
# CUDA_VISIBLE_DEVICES="0,1,2" accelerate launch --multi_gpu main.py  --model_name='samsum_dp' --save_ckpt  # 3_Final_SamsumDP
# CUDA_VISIBLE_DEVICES="0,1,2" accelerate launch --multi_gpu main.py  --model_name='dialogsum_dp' --save_ckpt  # 3_Final_DialogDP
# CUDA_VISIBLE_DEVICES="0,1,2" accelerate launch --multi_gpu main.py  --model_name='dialogsum_dp' --save_ckpt  # 3_Final_DialogDP

# cd ..
# CUDA_VISIBLE_DEVICES="0,1,2" accelerate launch --main_process_port 29501 --multi_gpu main.py  --model_name='xsum_stage_dp' --save_ckpt  # 3_Final_XSumDP
# CUDA_VISIBLE_DEVICES="0,1,2" accelerate launch --main_process_port 29501 --multi_gpu main.py  --model_name='xsum_dp' --save_ckpt  # 3_Final_XSumDP
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" accelerate launch --main_process_port 29501 --multi_gpu main.py  --model_name='xsum_stage_dp_full' --save_ckpt  # 3_Final_XSumDP
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" accelerate launch --main_process_port 29501 --multi_gpu main.py  --model_name='mrpc_stage_dp_full' --save_ckpt  # 3_Final_MRPCDP
# CUDA_VISIBLE_DEVICES="0,1,2,3,4" accelerate launch --main_process_port 29501 --multi_gpu main.py  --model_name='xsum_stage_dp_eps3' --save_ckpt  # 3_Final_MRPCDP


# Run with CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7" accelerate launch --multi_gpu main.py --save_ckpt

# Evaluate the Model on a test set
# cd stage2/train/
# python evaluate_model_sam.py  #4_EvalFinalSamSum #4_EvaluationFinalModel

# Evaluate with GPT-4
# cd evaluation/gpt4/
# python gpt4_evaluation.py

########################################################################################################################################################
# cd dp           # XSum_DP_Model
# python dp_summary_xsum.py
# python dp_summary_samsum.py  # SamSum_DP_Model
# python dp_summary_dialogsum.py   # DialogSum_DP_Model
# python evaluate_model.py  # EvaluateDPModel
# python dp_classification_sst2.py
# python dp_paraphrase_mrpc.py # Para_MRPC_
# python vanilla_paraphrase_mrpc.py #Paraphrasing_MRPC
# python dp_summary_pubmed.py

##### RUN DP MODELS
# cd dp  

### Classification Tasks
# python dp_classification_sst2.py --run_name "sst2_dp_epsilons_grad_20" --gradient_accumulation_steps 1 --target_epsilon 20
# python dp_classification_sst2.py --run_name "sst2_dp_epsilons_grad_8" --gradient_accumulation_steps 1 --target_epsilon 8
# python dp_classification_sst2.py --run_name "sst2_dp_epsilons_grad_1" --gradient_accumulation_steps 1 --target_epsilon 1

### Paraphrasing Tasks
# python dp_paraphrase_mrpc.py --run_name "mrpc_dp_epsilons_grad_20" --gradient_accumulation_steps 1 --target_epsilon 20
# python dp_paraphrase_mrpc.py --run_name "mrpc_dp_epsilons_grad_8" --gradient_accumulation_steps 1 --target_epsilon 8
# python dp_paraphrase_mrpc.py --run_name "mrpc_dp_epsilons_grad_1" --gradient_accumulation_steps 1 --target_epsilon 1

### Summarization Tasks
# python dp_summary.py --project_name DP_Summarization_XSum --dataset "xsum" --run_name "xsum_dp_epsilons_20" --target_epsilon 20  --epochs 2
# python dp_summary.py --project_name DP_Summarization_XSum --dataset_name "xsum" --run_name "xsum_dp_epsilons_8" --target_epsilon 8 --epochs 2
# python dp_summary.py --project_name DP_Summarization_XSum --dataset_name "xsum" --run_name "xsum_dp_epsilons_1" --target_epsilon 1 --epochs 2

# python dp_summary.py --project_name DP_Summarization_PubMed --dataset_name "pubmed" --run_name "pubmed_dp_epsilons_20" --target_epsilon 20
# python dp_summary.py --project_name DP_Summarization_PubMed --dataset_name "pubmed" --run_name "pubmed_dp_epsilons_8" --target_epsilon 8
# python dp_summary.py --project_name DP_Summarization_PubMed --dataset_name "pubmed" --run_name "pubmed_dp_epsilons_1" --target_epsilon 1

########################################################################################################################################################
# python -u vanilla_summary.py # VanillaModel
# python -u vanilla_summary_xsum.py  #VanillaXSum
# # python -u vanilla_summary_flan.py # VanillaModelFLAN
# python -u dp_summary_bart.py # DP_Dialog_FLAN

###################
# python -u evaluate_model.py         # EvaluateDPModel

# python -u evaluate_model.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/dp/samsum_results/samsum_dp' --run_name='samsum_dp'   # EvaluateSamSum_DP

# python -u evaluate_model.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/dp/dialogsum_results/dialogsum_dp' --run_name='dialogsum_dp'   # EvaluateDialogSum_DP

# python -u evaluate_model.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/stage_dp/stage_dp/models/samsum_stage_dp-1_1' --run_name='samsum_stage_dp-1' #EvaluateSamSum_StageDP

# python -u evaluate_model.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/stage_dp/stage_dp/models/dialogsum_stage_dp_1' --run_name='dialogsum_stage_dp-1' #EvaluateDialogSum_StageDP

# python -u evaluate_model.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/stage_dp/models/samsum_stage_dp_epochs_8' --run_name='samsum_stage_dp-8' #EvaluateSamSum_StageDP

# python -u evaluate_model_copy.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/stage_dp/models/dialogsum_stage_dp_8' --run_name='dialogsum_stage_dp-8' #EvaluateDialogSum_StageDP

#### FINAL MODELS
# python -u evaluate_model.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/stage2/train/gen_model/samsum_stage_dp' --run_name='samsum_stage_dp' #EvaluateFinalSamSum_DP

# python -u evaluate_model_copy.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/stage2/train/gen_model/dialogsum_stage_dp' --run_name='dialogsum_stage_dp' #EvaluateFinalDialogSum_DP


####

# python -u evaluate_model_xsum.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/dp/models/xsum_dp' --run_name='xsum_dp-80' # Evaluate_XSum_DP
# python -u evaluate_model_xsum.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/stage2/train/gen_model/xsum_stage_dp' --run_name='xsum_stage_dp-80' # Evaluate_Final_XSum
# python -u evaluate_model_xsum.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/dp/models/vanilla/checkpoint-2000' --run_name='vanilla-300' # Evaluate_XSum_Vanilla
# python -u evaluate_model_xsum.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/dp/models/vanilla/checkpoint-2000' --run_name='vanilla-300' # Evaluate_XSum_Vanilla

# python -u evaluate_model_xsum.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/dp/models/xsum_dp_epsilon3' --run_name='xsum_dp_epsilon3' # Evaluate_XSum_DP
# python -u evaluate_model_xsum.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/stage2/train/gen_model/xsum_stage_dp_eps3' --run_name='xsum_stage_dp_eps3' # Evaluate_Final_XSum

# python -u evaluate_model_xsum.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/dp/models/xsum_dp_ep_1' --run_name='xsum_dp_ep_1' # Evaluate_XSum_DP_Eps1
# cd ../..
# python -u evaluate_model_xsum.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/stage2/train/gen_model/xsum_stage_dp' --run_name='xsum_stage_dp_more_data4_epoch1' # Evaluate_Final_XSum
# python -u evaluate_model_xsum.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/stage2/train/gen_model/xsum_dp' --run_name='xsum_dp_distilled' # Evaluate_Final_XSum
# cd ..
# python -u evaluate_model_xsum.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/stage2/train/gen_model/xsum_stage_dp_full' --run_name='xsum_stage_dp_full' # Evaluate_Final_XSum 

# python -u evaluate_model_xsum.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/dp/models/xsum_dp_small_epsilon' --run_name='xsum_dp_small_epsilon' # Evaluate_XSum_DP_Eps1
# python -u evaluate_model_xsum.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/dp/models/xsum_dp_large_epsilon' --run_name='xsum_dp_large_epsilon' # Evaluate_XSum_DP_Eps1
# python -u evaluate_model_xsum.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/dp/models/xsum_dp_epsilon_8' --run_name='xsum_dp_epsilon_8' # Evaluate_XSum_DP_Eps1

# python -u evaluate_model_mrpc.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/dp/models/mrpc_dp_full' --run_name='mrpc_dp'  #Evaluate_DP_MRPC
# python -u evaluate_model_mrpc.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/dp/models/mrpc_vanilla_full/checkpoint-4000' --run_name='mrpc_vanilla' # Evaluate_Vanilla_MRPC
# python -u evaluate_model_mrpc.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/stage2/train/gen_model/mrpc_stage_dp_full' --run_name='mrpc_stage_dp_full' #Evaluate_Final_MRPC

# cd evaluation
# python -u evaluate_model_mrpc.py --model_path='/users/k/n/kngongiv/Research/clean/private_llm_generation/dp/models/mrpc_dp_epsilons_grad_8' --run_name='mrpc_dp_epsilon_8_no_grad'  #Evaluate_DP_MRPC
# python -u evaluate_model_mrpc.py --model_path='/users/k/n/kngongiv/Research/clean/private_llm_generation/dp/models/mrpc_dp_epsilons_grad_20' --run_name='mrpc_dp_epsilon_20_no_grad'  #Evaluate_DP_MRPC
# python -u evaluate_model_mrpc.py --model_path='/users/k/n/kngongiv/Research/clean/private_llm_generation/dp/models/mrpc_dp_epsilons_grad_1' --run_name='mrpc_dp_epsilon_1_no_grad'  #Evaluate_DP_MRPC

# python -u evaluate_model_xsum.py --model_path='/users/k/n/kngongiv/Research/clean/private_llm_generation/dp/models/xsum_dp_epsilons_8' --run_name='xsum_stagedp_epsilons_8'  #Evaluate_DP_XSUM

# python -u evaluate_model_pubmed.py --model_path='/users/k/n/kngongiv/Research/clean/private_llm_generation/dp/models/pubmed_dp_epsilons_8' --run_name='pubmed_dp_epsilons_8'  #Evaluate_DP_PubMed

#######
# python -u evaluate_model_mrpc.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/dp/models/mrpc_dp_small_epsilon_2' --run_name='mrpc_dp_small_epsilon_2'  #Evaluate_DP_MRPC
# python -u evaluate_model_mrpc.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/dp/models/mrpc_dp_large_epsilon_2' --run_name='mrpc_dp_large_epsilon_2'  #Evaluate_DP_MRPC
# python -u evaluate_model_mrpc.py --model_path='/users/k/n/kngongiv/Research/private_llm_generation/dialog/dp/models/mrpc_dp_epsilon_8' --run_name='mrpc_dp_epsilon_8'  #Evaluate_DP_MRPC

########## Evaluate with AlpacaEval
# cd evaluation/alpaca_eval/
# alpaca_eval evaluate --model_outputs='impossible_results_xsum.json' --reference_outputs='dp_results_xsum.json'
# alpaca_eval evaluate --model_outputs='impossible_more_xsum.json' --reference_outputs='dp_results_xsum.json'

# alpaca_eval evaluate --model_outputs='impossible_distil.json' --reference_outputs='dpsgd.json'

# alpaca_eval --model_outputs='impossible_distil.json' --reference_outputs='dpsgd.json'

# alpaca_eval --model_outputs='impossible_pubmeds.json' --reference_outputs='dp_pubmeds.json'


# alpaca_eval --model_outputs='impossible_distil.json' 
# # alpaca_eval --model_outputs='dpsgd.json' 
# # alpaca_eval --model_outputs='dpsgd.json' --reference_outputs='impossible_distil.json'

# METRICS
# XSUM
# cd evaluation/alpaca_eval/

# # Human Preference | Quality
# alpaca_eval evaluate --model_outputs='impossible_results_xsum.json' --reference_outputs='dp_results_xsum.json' --output_path='./results/xsum/quality/'

# # Fact Omission
# alpaca_eval evaluate --model_outputs='impossible_results_xsum.json' --reference_outputs='dp_results_xsum.json' --annotators_config='/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/metrics/fact_omission/configs.yaml' --output_path='./results/xsum/fact_omission/'

# # Coherence
# alpaca_eval evaluate --model_outputs='impossible_results_xsum.json' --reference_outputs='dp_results_xsum.json' --annotators_config='/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/metrics/coherence/configs.yaml' --output_path='./results/xsum/coherence/'

# # Consistency
# alpaca_eval evaluate --model_outputs='impossible_results_xsum.json' --reference_outputs='dp_results_xsum.json' --annotators_config='/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/metrics/consistency/configs.yaml' --output_path='./results/xsum/consistency/'

# # Relevance
# alpaca_eval evaluate --model_outputs='impossible_results_xsum.json' --reference_outputs='dp_results_xsum.json' --annotators_config='/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/metrics/relevance/configs.yaml' --output_path='./results/xsum/relevance/'

# # Fluency
# alpaca_eval evaluate --model_outputs='impossible_results_xsum.json' --reference_outputs='dp_results_xsum.json' --annotators_config='/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/metrics/fluency/configs.yaml' --output_path='./results/xsum/fluency/'

### 

#### PubMed
# cd evaluation/alpaca_eval/

# # # Human Preference | Quality
# alpaca_eval evaluate --model_outputs='impossible_pubmeds.json' --reference_outputs='dp_pubmeds.json' --output_path='./results/pubmed/quality/'

# # # Fact Omission
# alpaca_eval evaluate --model_outputs='impossible_pubmeds.json' --reference_outputs='dp_pubmeds.json' --annotators_config='/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/metrics/fact_omission/configs.yaml' --output_path='./results/pubmed/fact_omission/'

# # Coherence
# alpaca_eval evaluate --model_outputs='impossible_pubmeds.json' --reference_outputs='dp_pubmeds.json' --annotators_config='/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/metrics/coherence/configs.yaml' --output_path='./results/pubmed/coherence/'

# # Consistency
# alpaca_eval evaluate --model_outputs='impossible_pubmeds.json' --reference_outputs='dp_pubmeds.json' --annotators_config='/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/metrics/consistency/configs.yaml' --output_path='./results/pubmed/consistency/'

# Relevance
# alpaca_eval evaluate --model_outputs='impossible_pubmeds.json' --reference_outputs='dp_pubmeds.json' --annotators_config='/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/metrics/relevance/configs.yaml' --output_path='./results/pubmed/relevance/'

# Fluency
# alpaca_eval evaluate --model_outputs='impossible_pubmeds.json' --reference_outputs='dp_pubmeds.json' --annotators_config='/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/metrics/fluency/configs.yaml' --output_path='./results/pubmed/fluency/'

#### MPRC
# cd evaluation/alpaca_eval/

# # # Human Preference | Quality
# alpaca_eval evaluate --model_outputs='./data/impossible_mrpc.json' --reference_outputs='./data/dp_mrpc.json' --output_path='./results/mrpc/quality/'

# # # Fact Omission
# alpaca_eval evaluate --model_outputs='./data/impossible_mrpc.json' --reference_outputs='./data/dp_mrpc.json' --annotators_config='/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/metrics/fact_omission/configs.yaml' --output_path='./results/mrpc/fact_omission/'

# # Coherence
# alpaca_eval evaluate --model_outputs='./data/impossible_mrpc.json' --reference_outputs='./data/dp_mrpc.json' --annotators_config='/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/metrics/coherence/configs.yaml' --output_path='./results/mrpc/coherence/'

# # Consistency
# alpaca_eval evaluate --model_outputs='./data/impossible_mrpc.json' --reference_outputs='./data/dp_mrpc.json' --annotators_config='/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/metrics/consistency/configs.yaml' --output_path='./results/mrpc/consistency/'

# Relevance
# alpaca_eval evaluate --model_outputs='./data/impossible_mrpc.json' --reference_outputs='./data/dp_mrpc.json' --annotators_config='/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/metrics/relevance/configs.yaml' --output_path='./results/mrpc/relevance/'

# Fluency
# alpaca_eval evaluate --model_outputs='./data/impossible_mrpc.json' --reference_outputs='./data/dp_mrpc.json' --annotators_config='/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/metrics/fluency/configs.yaml' --output_path='./results/mrpc/fluency/'

#================================================================================
########## Evaluate with GEval => DP
# #### XSum
# cd evaluation/geval/

# # python gpt4_eval.py --prompt .\prompts\summeval\flu_detailed.txt --save_fp .\results\xsum\fluency_xsum.json --summeval_fp .\data\geval_dp_xsum.json --key 

# # Fluency
# python gpt4_eval.py --prompt ./prompts/summeval/flu_detailed.txt --save_fp ./results/dp_results/xsum/fluency_xsum.json --summeval_fp ./data/geval_dp_xsum.json --model gpt-4-0125-preview

# # # Coherence
# python gpt4_eval.py --prompt ./prompts/summeval/coh_detailed.txt --save_fp ./results/dp_results/xsum/coherence_xsum.json --summeval_fp ./data/geval_dp_xsum.json --model gpt-4-0125-preview

# # # Consistency
# python gpt4_eval.py --prompt ./prompts/summeval/con_detailed.txt --save_fp ./results/dp_results/xsum/consistency_xsum.json --summeval_fp ./data/geval_dp_xsum.json --model gpt-4-0125-preview

# # # # Relevance
# python gpt4_eval.py --prompt ./prompts/summeval/rel_detailed.txt --save_fp ./results/dp_results/xsum/relevance_xsum.json --summeval_fp ./data/geval_dp_xsum.json --model gpt-4-0125-preview


# #### PubMed
# cd evaluation/geval/

# # Fluency
# python gpt4_eval.py --prompt ./prompts/pubmed/flu_detailed.txt --save_fp ./results/dp_results/pubmed/fluency_pubmed.json --summeval_fp ./data/geval_dp_pubmed.json --model gpt-4-0125-preview

# # # Coherence
# python gpt4_eval.py --prompt ./prompts/pubmed/coh_detailed.txt --save_fp ./results/dp_results/pubmed/coherence_pubmed.json --summeval_fp ./data/geval_dp_pubmed.json --model gpt-4-0125-preview

# # # Consistency
# python gpt4_eval.py --prompt ./prompts/pubmed/con_detailed.txt --save_fp ./results/dp_results/pubmed/consistency_pubmed.json --summeval_fp ./data/geval_dp_pubmed.json --model gpt-4-0125-preview

# # # # Relevance
# python gpt4_eval.py --prompt ./prompts/pubmed/rel_detailed.txt --save_fp ./results/dp_results/pubmed/relevance_pubmed.json --summeval_fp ./data/geval_dp_pubmed.json --model gpt-4-0125-preview


#### MRPC
# cd evaluation/geval/

# # Fluency
# python gpt4_eval.py --prompt ./prompts/mrpc/flu_detailed.txt --save_fp ./results/dp_results/mrpc/fluency_mrpc.json --summeval_fp ./data/geval_dp_mrpc.json --model gpt-4-0125-preview

# # # Coherence
# python gpt4_eval.py --prompt ./prompts/mrpc/coh_detailed.txt --save_fp ./results/dp_results/mrpc/coherence_mrpc.json --summeval_fp ./data/geval_dp_mrpc.json --model gpt-4-0125-preview

# # # Consistency
# python gpt4_eval.py --prompt ./prompts/mrpc/con_detailed.txt --save_fp ./results/dp_results/mrpc/consistency_mrpc.json --summeval_fp ./data/geval_dp_mrpc.json --model gpt-4-0125-preview

# # # # Relevance
# python gpt4_eval.py --prompt ./prompts/mrpc/rel_detailed.txt --save_fp ./results/dp_results/mrpc/relevance_mrpc.json --summeval_fp ./data/geval_dp_mrpc.json --model gpt-4-0125-preview

## MetaEvaluaiton 
## cd evaluation/geval/
# python compute_metrics.py ./results/dp_results/xsum
# python compute_metrics.py ./results/dp_results/pubmed
# python compute_metrics.py ./results/dp_results/mrpc


###############################
########## Evaluate with GEval => IMPOSSIBLE
# #### XSum
# cd evaluation/geval/

# # python gpt4_eval.py --prompt .\prompts\summeval\flu_detailed.txt --save_fp .\results\xsum\fluency_xsum.json --summeval_fp .\data\geval_impossible_xsum.json --key 

# # # Fluency
# python gpt4_eval.py --prompt ./prompts/summeval/flu_detailed.txt --save_fp ./results/impossible_results/xsum/fluency_xsum.json --summeval_fp ./data/geval_impossible_xsum.json --model gpt-4-0125-preview

# # # Coherence
# python gpt4_eval.py --prompt ./prompts/summeval/coh_detailed.txt --save_fp ./results/impossible_results/xsum/coherence_xsum.json --summeval_fp ./data/geval_impossible_xsum.json --model gpt-4-0125-preview

# # # Consistency
# python gpt4_eval.py --prompt ./prompts/summeval/con_detailed.txt --save_fp ./results/impossible_results/xsum/consistency_xsum.json --summeval_fp ./data/geval_impossible_xsum.json --model gpt-4-0125-preview

# # # # Relevance
# python gpt4_eval.py --prompt ./prompts/summeval/rel_detailed.txt --save_fp ./results/impossible_results/xsum/relevance_xsum.json --summeval_fp ./data/geval_impossible_xsum.json --model gpt-4-0125-preview


# #### PubMed
# cd evaluation/geval/

# # Fluency
# python gpt4_eval.py --prompt ./prompts/pubmed/flu_detailed.txt --save_fp ./results/impossible_results/pubmed/fluency_pubmed.json --summeval_fp ./data/geval_impossible_pubmed.json --model gpt-4-0125-preview

# # # Coherence
# python gpt4_eval.py --prompt ./prompts/pubmed/coh_detailed.txt --save_fp ./results/impossible_results/pubmed/coherence_pubmed.json --summeval_fp ./data/geval_impossible_pubmed.json --model gpt-4-0125-preview

# # # Consistency
# python gpt4_eval.py --prompt ./prompts/pubmed/con_detailed.txt --save_fp ./results/impossible_results/pubmed/consistency_pubmed.json --summeval_fp ./data/geval_impossible_pubmed.json --model gpt-4-0125-preview

# # # # Relevance
# python gpt4_eval.py --prompt ./prompts/pubmed/rel_detailed.txt --save_fp ./results/impossible_results/pubmed/relevance_pubmed.json --summeval_fp ./data/geval_impossible_pubmed.json --model gpt-4-0125-preview


#### MRPC
# # cd evaluation/geval/

# # Fluency
# python gpt4_eval.py --prompt ./prompts/mrpc/flu_detailed.txt --save_fp ./results/impossible_results/mrpc/fluency_mrpc.json --summeval_fp ./data/geval_impossible_mrpc.json --model gpt-4-0125-preview

# # # Coherence
# python gpt4_eval.py --prompt ./prompts/mrpc/coh_detailed.txt --save_fp ./results/impossible_results/mrpc/coherence_mrpc.json --summeval_fp ./data/geval_impossible_mrpc.json --model gpt-4-0125-preview

# # # Consistency
# python gpt4_eval.py --prompt ./prompts/mrpc/con_detailed.txt --save_fp ./results/impossible_results/mrpc/consistency_mrpc.json --summeval_fp ./data/geval_impossible_mrpc.json --model gpt-4-0125-preview

# # # # Relevance
# python gpt4_eval.py --prompt ./prompts/mrpc/rel_detailed.txt --save_fp ./results/impossible_results/mrpc/relevance_mrpc.json --summeval_fp ./data/geval_impossible_mrpc.json --model gpt-4-0125-preview


## MetaEvaluaiton 
# cd evaluation/geval/
# python compute_metrics.py ./results/impossible_results/xsum
# python compute_metrics.py ./results/impossible_results/pubmed
# python compute_metrics.py ./results/impossible_results/mrpc

# # python meta_eval_summeval.py --input_fp .\results\gpt4_flu_detailed.json --dimension fluency
# python meta_eval_summeval.py --input_fp ./results/xsum/fluency_xsum.json --dimension fluency
# python meta_eval_summeval.py --input_fp ./results/xsum/coherence_xsum.json --dimension coherence
# python meta_eval_summeval.py --input_fp ./results/xsum/consistency_xsum.json --dimension consistency
# python meta_eval_summeval.py --input_fp ./results/xsum/relevance_xsum.json --dimension relevance


### Hallucination
# cd evaluation/
# python hallucination.py -d ./alpaca_eval/data/dp_results_xsum.json -o ./alpaca_eval/data/impossible_results_xsum.json -f ./hallucination_results/hallucinations_xsum.pdf -s ./hallucination_results/xsum_pii.json
# python hallucination.py -d ./alpaca_eval/data/dp_pubmeds.json -o ./alpaca_eval/data/impossible_pubmeds.json -f ./hallucination_results/hallucinations_pubmed.pdf -s ./hallucination_results/pubmed_pii.json
# python hallucination.py -d ./alpaca_eval/data/dp_mrpc.json -o ./alpaca_eval/data/impossible_mrpc.json -f ./hallucination_results/hallucinations_mrpc.pdf -s ./hallucination_results/mrpc_pii.json


# dpsgd_file = '/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/data/dp_results_xsum.json'
# ours_file = '/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/alpaca_eval/data/impossible_results_xsum.json'
# output_file = 'hallucinations_xsum.pdf'
# save_file = '/users/k/n/kngongiv/Research/private_llm_generation/dialog/evaluation/hallucination_results/xsum_pii.json'

