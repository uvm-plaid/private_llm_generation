# Decreasing Hallucinations in Differentially Private Language Models through Self-Distillation

This repo contains code for the paper "Decreasing Hallucinations in Differentially Private Language Models through Self-Distillation"

### Dependencies
Install all dependencies by running

```pip install -r requirements.txt```

### Reproducing Results in Paper

The following shows how to reproduce results in the paper. The 3 datasets used in this XSum, PubMed, and MRPC dataset. The DOMAIN can be "news" for xsum and mrpc  and "bio" for the pubmed dataset

### Running Results

#### 1. Phase 1 -  Distillation-Driven Synthesis:
Phase 1 generates high-quality input-output pairs with a small pre-trained model and fine-tunes an initial student model.

a. Generate initial contexts for dataset

```
cd stage1/summary/ 
python3 main_input.py --device_id={DEVICE_ID} --domain={DOMAIN}
```
b. Generate and filter candidate pairs 
The model `gpt2-xl ` is used for the news domain and `BioGPT-large` for the bio domain.
```
python main_output.py --device_id=0 --con_domain={DOMAIN} --con_model_name={MODEL_NAME} --shard_start=0 --shard_size=100000
```
Combine all generated datasets and split into train and validation sets

```
cd stage1/train/data/
python process.py --device_id={DEVICE_ID}
```

c. Train Student Model 

```
cd stage1/train/data/

CUDA_VISIBLE_DEVICES="0" accelerate launch --multi_gpu main.py --save_ckpt
```

#### 2. Phase 2 - DP finetuning
Finetune initial student model on private dataset with using DPSGD.

```
cd stage_dp
python dp_summary_stagedp.py --project_name {PROJECT_NAME} --dataset_name {DATASET_NAME} --run_name {RUN_NAME} --target_epsilon 8 --student_model_dir {STUDENT_MODEL_PATH}
```

#### 3. Phase 3 - Self-Distillation

a. Generate initial contexts for dataset

```
cd stage2/generate_and_filter/
python3 main_input.py --device_id={DEVICE_ID} --domain={DOMAIN}
```
b. Generate and filter candidate pairs 

```
python main_output.py --device_id=0 --con_domain={DOMAIN} --con_model_name={DPSGD_MODEL_NAME} --shard_start=0 --shard_size=100045

```
Combine all generated datasets and split into train and validation sets

```
cd stage2/train/data/
python process.py --device_id={DEVICE_ID}
```

c. Train Final Model 

```
cd stage1/train/data/

CUDA_VISIBLE_DEVICES="0,1,2" accelerate launch --main_process_port 29501 --multi_gpu main.py  --model_name={DPSGD_MODEL_NAME} --save_ckpt 

```

### Train DPSGD Baseline 

Train dpsgd baseline for comparison.

```
cd dp
python dp_summary_xsum.py
```


### Evaluation
a. Automatic Metics
Generate summaries on test set and compute automatic metrics like rouge, blue scores etc

```
python -u evaluate_model.py --model_path={FINAL_MODEL} --run_name={RUN_NAME}  

```

b. Evaluating model using AlpacaEval
Compare dpsgd baseline results with ours using AlpacaEval
```
cd evaluation/alpaca_eval/
alpaca_eval evaluate --model_outputs={OUR_RESULTS.json} --reference_outputs={DP_RESULTS.json}

```

