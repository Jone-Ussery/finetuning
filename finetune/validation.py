# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# Tools for performing validation over models.

import math
import torch
import typing
import constants
import traceback
import bittensor as bt
import transformers


def iswin(loss_i, loss_j, block_i, block_j):
    """
    Determines the winner between two models based on the epsilon adjusted loss.

    Parameters:
        loss_i (float): Loss of uid i on batch
        loss_j (float): Loss of uid j on batch.
        block_i (int): Block of uid i.
        block_j (int): Block of uid j.
    Returns:
        bool: True if loss i is better, False otherwise.
    """
    # Adjust loss based on timestamp and pretrain epsilon
    loss_i = (1 - constants.timestamp_epsilon) * loss_i if block_i < block_j else loss_i
    loss_j = (1 - constants.timestamp_epsilon) * loss_j if block_j < block_i else loss_j
    return loss_i < loss_j


def compute_wins(
    uids: typing.List[int],
    losses_per_uid: typing.Dict[int, typing.List[float]],
    uid_to_block: typing.Dict[int, int],
):
    """
    Computes the wins and win rate for each model based on loss comparison.

    Parameters:
        uids (list): A list of uids to compare.
        losses_per_uid (dict): A dictionary of losses for each uid by batch.
        batches (List): A list of data batches.
        uid_to_block (dict): A dictionary of blocks for each uid.

    Returns:
        tuple: A tuple containing two dictionaries, one for wins and one for win rates.
    """
    wins = {uid: 0 for uid in uids}
    win_rate = {uid: 0 for uid in uids}
    for i, uid_i in enumerate(uids):
        total_matches = 0
        block_i = uid_to_block[uid_i]
        for j, uid_j in enumerate(uids):
            if i == j:
                continue
            block_j = uid_to_block[uid_j]
            batches_i = len(losses_per_uid[uid_i])
            batches_j = len(losses_per_uid[uid_j])
            for batch_idx in range(0, min(batches_i, batches_j)):
                loss_i = losses_per_uid[uid_i][batch_idx]
                loss_j = losses_per_uid[uid_j][batch_idx]
                wins[uid_i] += 1 if iswin(loss_i, loss_j, block_i, block_j) else 0
                total_matches += 1
        # Calculate win rate for uid i
        win_rate[uid_i] = wins[uid_i] / total_matches if total_matches > 0 else 0

    return wins, win_rate


def compute_losses(
    model, batches: typing.List[typing.Tuple[torch.Tensor, int]], device: str
) -> typing.List[float]:
    """
    Computes the losses for a given model on provided batches.

    Parameters:
        model (torch.nn.Module): The model for which losses are to be computed.
        batches (dict): A list of batches and the associated lengths of the "prompt" section.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').

    Returns:
        dict: A dictionary with page indices as keys and lists of loss values as values.
    """
    # Iterate over each page and corresponding batches
    losses = []
    with torch.inference_mode():
        model.to(device)
        model.eval()
        for inputs, prompt_len in batches:
            try:
                inputs = inputs.to(device)
                labels = inputs.clone()
                labels[:, :prompt_len] = -100  # Only calculate loss on response
                outputs = model(inputs, labels=labels)
                loss = outputs.loss.item()  # Extract scalar loss value
                losses.append(loss)
            except Exception as e:
                bt.logging.error(f"Exception occurred in loss computation: {e}")
                traceback.print_exc()  # Print the stack trace
                losses.append(math.inf)  # Use infinity to indicate failure

    return losses


def generate_output(
    model,
    input_ids: torch.Tensor,
    generation_config: transformers.GenerationConfig,
    device: str,
    tokenizer: transformers.PreTrainedTokenizer,
) -> str:
    """
    Generates the tokenized output for a model given a tokenized input and generation config.

    Args:
        model (torch.nn.Module): The model for which losses are to be computed.
        input_ids (torch.Tensor): Input tokens to generate a response to.
        generation_config (transformers.GenerationConfig): Configuration parameters for generating output.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to tokenize the output with before returning.

    Returns:
        str: Generated tokenized output from the model.
    """
    with torch.inference_mode():
        model.to(device)
        model.eval()
        input_ids = input_ids.to(device)
        output = model.generate(
            input_ids=input_ids, generation_config=generation_config
        )
        response = tokenizer.decode(
            output[0][len(input_ids[0]) :], skip_special_tokens=True
        )
        return response
