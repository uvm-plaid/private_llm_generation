
# CODE: https://github.com/pytorch/opacus/blob/main/opacus/accountants/utils.py#L23
from typing import Optional
from opacus.accountants import create_accountant
MAX_SIGMA = 1e6

def get_noise_multiplier(
    *,
    target_epsilon: float,
    target_delta: float,
    sample_rate: float,
    epochs: Optional[int] = None,
    steps: Optional[int] = None,
    accountant: str = "rdp",
    epsilon_tolerance: float = 0.01,
    **kwargs,
) -> float:
    r"""
    Computes the noise level sigma to reach a total budget of (target_epsilon, target_delta)
    at the end of epochs, with a given sample_rate

    Args:
        target_epsilon: the privacy budget's epsilon
        target_delta: the privacy budget's delta
        sample_rate: the sampling rate (usually batch_size / n_data)
        epochs: the number of epochs to run
        steps: number of steps to run
        accountant: accounting mechanism used to estimate epsilon
        epsilon_tolerance: precision for the binary search
    Returns:
        The noise level sigma to ensure privacy budget of (target_epsilon, target_delta)
    """
    if (steps is None) == (epochs is None):
        raise ValueError(
            "get_noise_multiplier takes as input EITHER a number of steps or a number of epochs"
        )
    if steps is None:
        steps = int(epochs / sample_rate)

    eps_high = float("inf")
    accountant = create_accountant(mechanism=accountant)

    sigma_low, sigma_high = 0, 10
    while eps_high > target_epsilon:
        sigma_high = 2 * sigma_high
        accountant.history = [(sigma_high, sample_rate, steps)]
        eps_high = accountant.get_epsilon(delta=target_delta, **kwargs)
        if sigma_high > MAX_SIGMA:
            raise ValueError("The privacy budget is too low.")

    while target_epsilon - eps_high > epsilon_tolerance:
        sigma = (sigma_low + sigma_high) / 2
        accountant.history = [(sigma, sample_rate, steps)]
        eps = accountant.get_epsilon(delta=target_delta, **kwargs)

        if eps < target_epsilon:
            sigma_high = sigma
            eps_high = eps
        else:
            sigma_low = sigma

    return sigma_high


from datasets import load_dataset
# load xsum
xsum_train_data = load_dataset("xsum", split="train")
# load mrpc
mrpc_dataset = load_dataset("glue", "mrpc")
mrpc_train_data = mrpc_dataset['train']
# load pubmed 

batch_size = 4
epochs = 4
target_epsilon = 8

xsum_target_delta = 1/(10 * len(xsum_train_data))
xsum_sample_rate = batch_size/len(xsum_train_data)  # self.sampling_rate = self.batch_size / self.data_size

mrpc_target_delta = 1/(10 * len(mrpc_train_data))
mrpc_sample_rate = batch_size/len(mrpc_train_data)  

xsum_noise_multiplier = get_noise_multiplier(target_epsilon=target_epsilon, target_delta=xsum_target_delta, sample_rate=xsum_sample_rate, epochs=epochs, accountant='rdp') 
mrpc_noise_multiplier = get_noise_multiplier(target_epsilon=target_epsilon, target_delta=mrpc_target_delta, sample_rate=mrpc_sample_rate, epochs=epochs, accountant='rdp') 


print(f"XSum Dataset: \n Noise Multiplier: {xsum_noise_multiplier},  Epsilon: {target_epsilon} , Delta: {xsum_target_delta}, Sample Rate: {xsum_sample_rate},  Epochs: {epochs}, Batch Size: {batch_size}")
print(f"\nMRPC Dataset: \n Noise Multiplier: {mrpc_noise_multiplier},  Epsilon: {target_epsilon} , Delta: {mrpc_target_delta}, Sample Rate: {mrpc_sample_rate},  Epochs: {epochs}, Batch Size: {batch_size}")



# NOTE:
# If you train for 5 epochs, save the model, and then load and train it for another 5 epochs using the same noise multiplier calculated for a total of 10 epochs, you should still be within your desired privacy budget of (\epsilon = 8, \delta = 1 \times 10^{-5}).
# It is crucial that the total number of epochs does not exceed the original plan (10 epochs) for which the noise multiplier was calculated. If you train for more epochs, you would need to recalculate a new noise multiplier to fit the extended training within the desired privacy bounds.



# XSum Dataset: 
#  Noise Multiplier: 0.369110107421875,  Epsilon: 8 , Delta: 4.90087970790757e-07, Sample Rate: 1.960351883163028e-05,  Epochs: 4, Batch Size: 4

# MRPC Dataset: 
#  Noise Multiplier: 0.43853759765625,  Epsilon: 8 , Delta: 2.7262813522355506e-05, Sample Rate: 0.0010905125408942203,  Epochs: 4, Batch Size: 4

# PubMed Dataset: 
#  Noise Multiplier: 0.375823974609375,  Epsilon: 8 , Delta: 8.33861445582202e-07, Sample Rate: 0.0010905125408942203,  Epochs: 4, Batch Size: 4

