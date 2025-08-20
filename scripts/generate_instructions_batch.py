import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import json
from datasets import load_dataset
import random
from utils.generate_instructions import generate_instruction_sample

def get_random_setting():
    ds = load_dataset("SALT-NLP/NormBank")["train"].filter(lambda x: x["norm"] != "taboo")
    settings = ds['setting']
    return random.choice(settings)

def get_random_conversation_type():
    return random.choice(["monologue", "dialogue"])

def main(args):
    with open(args.metadata) as f:
        metadata = json.load(f) 

    category = metadata["category"]
    task_description = metadata["task_description"]
    prompt_path = metadata["prompt_file"]
    out_dir = metadata["data_folder"]
    os.makedirs(out_dir, exist_ok=True)
    model = args.model
    temperature = args.temperature

    # Loop
    generated = 0 
    while generated < args.num_samples:

        # Use the provided setting if given, else pick a random one
        setting = args.setting
        if setting is None or setting == "":
            setting = get_random_setting()

        # Use provided conversation type as-is (no random selection here)
        conversation_type = args.conversation_type

        sample, sample_id = generate_instruction_sample(
            prompt_path,
            category,
            task_description,
            setting,
            out_dir,
            model,
            temperature,
            conversation_type=conversation_type,
        )
        if sample is not None:
            generated += 1
            print(f"Generated {generated} samples")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--setting", type=str, default=None, help="Context setting for the prompt. If not provided, a random one is chosen.")
    parser.add_argument("--model", type=str, default="gpt-4o")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--conversation_type", type=str, choices=["monologue", "dialogue"], default=None, help="Optionally force conversation type; if omitted, the prompt may specify <<CONVERSATION_TYPE>> or the model may choose.")
    args = parser.parse_args()
    main(args)


# USAGE:
# python scripts/generate_instructions_batch.py --metadata categories/lexical_stress_shift.json --num_samples 100 --model gpt-4o 






