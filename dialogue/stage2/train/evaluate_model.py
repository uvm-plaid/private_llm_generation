import os
import torch
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast
from argparse import Namespace
import wandb

from dataset import CTGDataset
from util import init_tokenizer_and_model, set_seed

# Assume evaluate function is imported here
from main import evaluate

def load_model(model_path, device):
    """
    Load the saved model from a specified path.
    """
    # model = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location=device)
    config = torch.load(os.path.join(model_path, 'config.json'), map_location=device)
    model.config.update(config)
    return model

def load_test_data(tokenizer, test_filename, batch_size):
    """
    Load test data into a DataLoader.
    """
    test_dataset = CTGDataset.from_file(tokenizer, test_filename)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 collate_fn=CTGDataset.collate_fn, num_workers=4)
    return test_dataloader

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)  # Ensure reproducibility
    
    tokenizer, _ = init_tokenizer_and_model(args)
    model = load_model(args.model_dir, device).to(device)
    model.eval()  # Set the model to evaluation mode

    test_dataloader = load_test_data(tokenizer, args.test_filename, args.test_batch_size)

    wandb_table = wandb.Table(columns=["Epoch", "Original Text", "Original Label", "Predicted Text"])
    args.accelerator = Namespace(is_main_process=True, device=device)  # Simplified for this example

    # Evaluate the model
    _, generation_table = evaluate(model, test_dataloader, tokenizer, args, epoch=0, wandb_table=wandb_table)

    # Log the results to wandb, if desired
    wandb.log({"Test Generation": generation_table})

if __name__ == "__main__":
    args = Namespace(
        seed=42,
        model_dir='/users/k/n/kngongiv/Research/impossible_kd/SummKD-submission/stage2/train/gen_model/drawn-disco-9',
        test_filename='path_to_test_data.txt',
        test_batch_size=16,
        # Add other necessary arguments here
    )
    main(args)
