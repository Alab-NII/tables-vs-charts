# prompts/claim_2_prompt_template.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import format_caption_and_table
import json


def build_zero_shot_cot_prompt(claim, table=None, image_path=None, image_caption=None):
    prompt_parts = []

    if image_path:
        prompt_parts.append({
            "type": "image",
            "image": image_path
        })

    instruction = "Use the provided "
    if table and image_path:
        instruction += "table and image"
    elif table:
        instruction += "table"
    elif image_path:
        instruction += "image"
    else:
        instruction += "information"

    instruction += f", predict the label for this claim: '{claim}'; the label can be Supported or Refuted. Think step by step before answering. Please format your final answer within brackets as follows: <ans> YOUR ANSWER </ans>"

    if table:
        instruction = f"Table Information:\n{table}\n\n" + instruction


    # Add image caption if available
    if image_caption:
        instruction = f"Image Caption:\n{image_caption}\n\n" + instruction

    prompt_parts.append({
        "type": "text",
        "text": instruction
    })

    return prompt_parts




