"""
This is copy from the On-Demand-IE https://github.com/yzjiao/On-Demand-IE/tree/main

A dedicated helper to manage templates and prompt building.
"""

import json
import os.path as osp
from typing import Union


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template_name: str = "", verbose: bool = False):
        self._verbose = verbose
        if not template_name:
            # Enforce the default here, so the constructor can be called with '' and will not break.
            template_name = "alpaca"
        file_name = osp.join("templates", f"{template_name}.json")
        if not osp.exists(file_name):
            raise ValueError(f"Can't read {file_name}")
        with open(file_name) as fp:
            self.template = json.load(fp)
        if self._verbose:
            print(
                f"Using prompt template {template_name}: {self.template['description']}"
            )

    def generate_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        label: Union[None, str] = None,
    ) -> str:
        # returns the full prompt from instruction and optional input
        # if a label (=response, =output) is provided, it's also appended.
        if input:
            res = self.template["prompt_input"].format(
                instruction=instruction, input=input
            )
        else:
            res = self.template["prompt_no_input"].format(
                instruction=instruction
            )
        if label:
            res = f"{res}{label}"
        if self._verbose:
            print(res)
        return res
    
    def generate_chat_prompt(
        self,
        messages: Union[None, str] = None,
    ) -> str:
        # returns the full prompt for a conversation history
        if len(messages) == 0:
            raise ValueError('Messages field is empty.')

        message_text = ""
        for message in messages:
            if message["role"] == "system":
                message_text += "<|system|>\n" + message["content"].strip() + "\n\n"
            elif message["role"] == "user":
                message_text += "<|user|>\n" + message["content"].strip() + "\n\n"
            elif message["role"] == "assistant":
                message_text += "<|assistant|>\n" + message["content"].strip() + "\n\n"
            else:
                raise ValueError("Invalid role: {}".format(message["role"]))
        return message_text

    def get_response(self, output: str, use_chat_prompt=False) -> str:
        if use_chat_prompt:
            return output.split('<|assistant|>\n')[-1].strip()
        else:
            return output.split(self.template["response_split"])[1].strip()
