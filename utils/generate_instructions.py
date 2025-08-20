import os
import json
import hashlib
from dotenv import load_dotenv
load_dotenv()
import openai
import re


# 1. LOAD PROMPT TEMPLATE AND FORMAT WITH SETTING
def load_prompt_template(prompt_path, category, task_description, setting, conversation_type=None):
    with open(prompt_path) as f:
        template = f.read()

    # Replace placeholders with actual values
    template = template.replace("<<CATEGORY>>", category)
    template = template.replace("<<TASK_DESCRIPTION>>", task_description)
    template = template.replace("<<SETTING>>", setting)
    if conversation_type:
        template = template.replace("<<CONVERSATION_TYPE>>", conversation_type)

    return template

# 2. GENERATE GPT SAMPLE
def generate_gpt_response(prompt, model="gpt-4o", temperature=0.7):
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant for generating structured dataset samples."},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature
    )
    return response.choices[0].message.content.strip()

# 3. PARSE JSON RESPONSE
def parse_json_response(response):

    try:
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if not match:
            raise ValueError("No JSON object found.")

        json_str = match.group(0)
        # Load only the first JSON object, allow trailing text
        decoder = json.JSONDecoder()
        parsed, _ = decoder.raw_decode(json_str)

        return parsed

    except Exception as e:  
        print("❌ Failed to parse GPT output:", e)
        print("↪ Response was:\n", response[:500], "...\n")  # preview the output
        return None

# 4. GET NEXT SAMPLE ID
def get_next_sample_id(data_dir):
    existing = [
        d for d in os.listdir(data_dir)
        if d.startswith("sample_") and os.path.isdir(os.path.join(data_dir, d))
    ]
    if not existing:
        return 1
    ids = [int(d.split("_")[1]) for d in existing]
    return max(ids) + 1

# 5. SAVE SAMPLE JSON
def save_sample_json(sample, out_dir, sample_id):
    sample_folder = os.path.join(out_dir, f"sample_{sample_id:04d}")
    os.makedirs(sample_folder, exist_ok=True)
    out_path = os.path.join(sample_folder, "meta.json")
    with open(out_path, "w") as f:
        json.dump(sample, f, indent=2)
    print(f"✅ Saved: {out_path}")

# MAIN: GENERATE INSTRUCTION SAMPLE
def generate_instruction_sample(prompt_path, category, task_description, setting, out_dir, model="gpt-4o", temperature=0.7, conversation_type=None):
    prompt = load_prompt_template(prompt_path, category, task_description, setting, conversation_type)
    response = generate_gpt_response(prompt, model, temperature)
    sample = parse_json_response(response)
    sample_id = get_next_sample_id(out_dir)
    save_sample_json(sample, out_dir, sample_id)
    return sample, sample_id

# USAGE: 
# python utils/generate_instructions.py --prompt_path prompts/lexical_stress_shift_improved.txt --category lexical_stress_shift --task_description "Stress Shift" --out_dir data/lexical_stress_shift






