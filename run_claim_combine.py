import os
import sys
import re
import json
from tqdm import tqdm
from tasks.claim_2.claim_runner import run_claim_task
from models.llama_vl import LlamaVLModel
from models.qwen_vl import QwenVLModel
from models.llava import LlavaModel
from models.intern_vl import InternVLModel
from models.intern_vl_multi import InternVL_MULTI_Model
from utils import format_caption_and_table, extract_answer

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

model_name = "internvl3-1b" # llama-3.2-11b qwen-2.5-vl-32b
prompt_type = "zeroshot"
table_format = "pipe_tagging"
chart_type = "basic_chart" 

model_registry = {
    "qwen-2.5-vl-32b": lambda: QwenVLModel(model_name="Qwen2.5-VL-32B"),
    "llama-3.2-11b": lambda: LlamaVLModel(model_name="Llama-3.2-11B-Vision"),
    "llava-v1.6-mistral-7b": lambda: LlavaModel(model_name="llava-v1.6-mistral-7b"),
    "internvl3-1b": lambda: InternVLModel(model_name="InternVL3-1B"),
    "internvl3-8b": lambda: InternVLModel(model_name="InternVL3-8B"),
    "internvl3-14b": lambda: InternVLModel(model_name="InternVL3-14B"),
    "internvl3-38b": lambda: InternVL_MULTI_Model(model_name="InternVL3-38B")
}

model = model_registry[model_name]()

OUTPUT_DIR = f"outputs/claim_task_combine"
os.makedirs(OUTPUT_DIR, exist_ok=True)


kwargs = {
    "temperature": 0,
    "max_tokens": 1024
}

dataset_file = "data/scitab_align_plus/data_img.json"
with open(dataset_file, "r", encoding="utf-8") as f:
    samples = json.load(f)
    # samples = samples[0:5]

results = []
print(len(samples))
for item in tqdm(samples):
    # print(item["id"])
    image_path = item[f"{chart_type}"] 
    table = format_caption_and_table(item["table_column_names"], item["table_content_values"], item["table_caption"], type_=table_format)
    #
    #
    # image_caption = re.sub(r'\bTable (?=\d)', 'Figure ', item["table_caption"])
    image_caption = None
    claim = re.sub(r'\bTable (?=\d)', 'Figure ', item["claim"])
    prompt, response = run_claim_task(claim, table, image_path, image_caption, model, shots=prompt_type, **kwargs)
    #
    results.append({
        "id": item["id"],
        "claim": claim,
        "label": item["label"],      
        "pred_label": extract_answer(response),
        "generated_response": response,
        "image_caption": image_caption,
        "paper_id": item["paper_id"],
        "table_id": item["table_id"],
        "user_prompt": prompt
    })

output_path = os.path.join(
    OUTPUT_DIR, f"{model_name.lower()}_{prompt_type}_{chart_type}.json")
print(f"Saving output to: {output_path}")
with open(output_path, "w", encoding="utf-8") as out_f:
    json.dump(results, out_f, indent=2, ensure_ascii=False)


