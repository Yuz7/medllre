import os
import sys
import fire
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
from tqdm import tqdm
from pathlib import Path
import json, jsonlines
from datetime import datetime
from datasets import load_dataset

def main1(
    load_8bit: bool = True,
    base_model: str = '/PATH/TO/MODEL',
    lora_weights: str = '/PATH/TO/FINETUNED/WEIGHTS',
    test_data_path: str = "/PATH/TO/TESTDATA/",
    results_dir: str = "",
    fold: int=0,
    trigger: str = 'cured',
):

    STOP_TOKEN='\n\nEND\n\n'

    starttime = datetime.strptime(str(datetime.now()),"%Y-%m-%d %H:%M:%S.%f")

    if not os.path.exists(results_dir):
        print("mkdir...",str(results_dir))
        os.makedirs(results_dir)

    val_data = load_dataset("json", data_files = test_data_path)
    val_data = val_data['train']

    tokenizer = LlamaTokenizer.from_pretrained(base_model)
    
    model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            device_map="auto",
        )
    model = PeftModel.from_pretrained(
            model,
            lora_weights,
            device_map="auto",
        )
    
    model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
    model.config.bos_token_id = 1
    model.config.eos_token_id = 2

    with open(results_dir+f'/run_{trigger}_{fold}.jsonl','w') as outfile:

        for d in tqdm(val_data):
            prompt = ""
            for message in d['messages']:
                if message["role"] == "system":
                    prompt += "<|system|>\n" + message["content"].strip() + "\n\n"
                elif message["role"] == "user":
                    prompt += "<|user|>\n" + message["content"].strip() + "\n\n"
                elif message["role"] == "assistant":
                    prompt += "<|assistant|>\n"
                else:
                    raise ValueError("Invalid role: {}".format(message["role"]))
                
            model_input = tokenizer(prompt,return_tensors='pt').to('cuda')
            model.eval()
            with torch.no_grad():
                response = tokenizer.decode(model.generate(**model_input,do_sample=False, max_new_tokens=2048)[0], skip_special_tokens=True)
                print(response)
                response = response.replace(prompt,"")
                if response.endswith(STOP_TOKEN):
                    response = response.replace(STOP_TOKEN,"")
            model.train()

            d['completion']=response #Note that this is not GPT3 completion but made it consistent for evaluation code

            json.dump(d,outfile)
            outfile.write('\n')
    endtime = datetime.strptime(str(datetime.now()),"%Y-%m-%d %H:%M:%S.%f")
    print("Time consumed ",endtime-starttime)

if __name__ == "__main__":
    fire.Fire(main1)
