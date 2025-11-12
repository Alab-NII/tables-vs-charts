import json
import sys
import ast
from evaluation.eval_claim import evaluate_claim_verification


def load_json(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_prediction(label):
    if not label or "row" in label or "-" in label:
        return []
    try:
        return ast.literal_eval(label)
    except (SyntaxError, ValueError):
        return []

def parse_ground_truth(label):
    if isinstance(label, str):
        try:
            return ast.literal_eval(label)
        except (SyntaxError, ValueError):
            return []
    return label

def evaluate_claim_task(models, prompt_types, base_path):
    print("Running Claim Verification Task Evaluation")
    for prompt_type in prompt_types:
        print(f"\nPrompt type: {prompt_type}")
        for model_name in models:
            input_path = f"{base_path}/{model_name}_{prompt_type}.json"
            data = load_json(input_path)

            predictions = [item["pred_label"] for item in data]
            ground_truth = [item["label"] for item in data]

            # For Claim task ===========================
            scores = evaluate_claim_verification(predictions, ground_truth)

            print(scores["macro_f1"])



def main():
    models = ["qwen-2.5-vl-3b", "qwen-2.5-vl-7b", "qwen-2.5-vl-32b", "qwen-2.5-vl-72b", "llama-3.2-11b", "llava-v1.6-mistral-7b", "llava-v1.6-vicuna-13b", "llava-v1.6-34b", "internvl3-1b", "internvl3-8b", "internvl3-14b", "internvl3-38b"] #                                 
    
    prompt_types = ["zeroshot"] 
    base_path = "outputs/claim_task_combine" # claim_task_table claim_task_table_162 claim_task_img claim_task_combine
    
    if "claim_task_img" in base_path and "mimic" not in base_path:
        prompt_types = ["zeroshot_basic_chart", "zeroshot_chart_symbol", "zeroshot_line_chart", "zeroshot_swapped_chart"]

    if "combine" in base_path:
        prompt_types = ["zeroshot_basic_chart"]

    evaluate_claim_task(models, prompt_types, base_path)


if __name__ == "__main__":
    main()
