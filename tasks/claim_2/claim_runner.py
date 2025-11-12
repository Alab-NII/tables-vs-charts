# tasks/claim_2/claim_runner.py

from prompts.claim_2_prompt_template import build_zero_shot_cot_prompt
from models.base_model import BaseModel


def run_claim_task(input: str, table: str, image_url: str, image_caption: str, model: BaseModel, shots="", **kwargs) -> str:
    prompt = build_zero_shot_cot_prompt(input, table=table, image_path=image_url, image_caption=image_caption)
    return model.call(prompt)