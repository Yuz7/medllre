import copy
import json
import os
import torch

from sentencepiece import SentencePieceProcessor
from torch.utils.data import Dataset
from utils.Prompt import Prompter

from datasets import load_dataset

def tokenize(prompt, tokenizer, cutoff_len = 2048, add_eos_token=True):
    # there's probably a way to do this with the tokenizer settings
    # but again, gotta move fast
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=cutoff_len,
        padding=False,
        return_tensors=None,
    )
    if (
        result["input_ids"][-1] != tokenizer.eos_token_id
        and len(result["input_ids"]) < cutoff_len
        and add_eos_token
    ):
        result["input_ids"].append(tokenizer.eos_token_id)
        result["attention_mask"].append(1)

    result["labels"] = result["input_ids"].copy()

    return result
    
def generate_and_tokenize_prompt(message, prompter, tokenizer, cutoff_len = 2048):
    full_prompt = prompter.generate_chat_prompt(
        message['messages']
    )
    
    tokenized_full_prompt = tokenize(full_prompt, tokenizer, cutoff_len)

    user_prompt = prompter.generate_chat_prompt(
        message['messages'][:-1]
    )
    user_prompt += "<|assistant|>\n"
    
    tokenized_user_prompt = tokenize(
        user_prompt, tokenizer, cutoff_len, add_eos_token=False
    )
    user_prompt_len = len(tokenized_user_prompt["input_ids"])

    # if add_eos_token:
    #     user_prompt_len -= 1

    tokenized_full_prompt["labels"] = [
        -100
    ] * user_prompt_len + tokenized_full_prompt["labels"][
        user_prompt_len:
    ]  # could be sped up, probably

    return tokenized_full_prompt

class MedREDataset(Dataset):
    def __init__(self, data_path, tokenizer, split_name, cutoff_len=2048):
        #self.data = json.load(open(dataset_config.data_path))
        
        self.prompter = Prompter()
        if split_name == "train":
            self.data = load_dataset("json", data_files = data_path+"/train.jsonl")
        else:
            self.data = load_dataset("json", data_files = data_path+"/val.jsonl")

        self.data = self.data['train']
        self.cutoff_len = cutoff_len
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]

        return generate_and_tokenize_prompt(item,self.prompter, self.tokenizer, self.cutoff_len)
