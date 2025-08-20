import os
import json
import openai
from dotenv import load_dotenv
load_dotenv()

from datasets import load_dataset
from huggingface_hub import InferenceClient

# 1. TTS CALL
def tts_call(text, voice, instructions, out_path):
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
        instructions=instructions,
        response_format="wav"
    )
    with open(out_path, "wb") as f:
        f.write(response.content)
    print(f"✅ Saved: {out_path}")


# 2. GENERATE AUDIO SAMPLE
def generate_audio_sample(sample_dir, input_voice=None, option_voice=None):
    meta_path = os.path.join(sample_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"No meta.json found in: {sample_dir}")
    with open(meta_path) as f:
        data = json.load(f)

    input_text = data["input_text"]
    input_instructions = data["input_instructions"] 
    options = data["options_text"]
    option_instructions = data["options_instructions"]

    assert len(input_instructions) == len(options) == len(option_instructions), \
        "Mismatched input/options/instructions lengths"

    # Resolve voices cleanly using helper
    conversation_type = data.get("conversation_type", None)
    input_voice, option_voice = _resolve_voices(conversation_type, input_voice, option_voice)

    # Persist chosen voices back to meta.json so they can be retrieved later
    voices_changed = (
        data.get("voice_target") != input_voice or data.get("voice_option") != option_voice
    )
    if voices_changed:
        data["voice_target"] = input_voice
        data["voice_option"] = option_voice
        try:
            with open(meta_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not write voices to meta.json for {sample_dir}: {e}")

    for i, (instr, opt, opt_instr) in enumerate(zip(input_instructions, options, option_instructions)):
        input_out_path = os.path.join(sample_dir, f"target{i+1}.wav")
        option_out_path = os.path.join(sample_dir, f"option{i+1}.wav")

        tts_call(input_text, input_voice, instr, input_out_path)
        tts_call(opt, option_voice, opt_instr, option_out_path)


# HELPER FUNCTIONS

# Available TTS voices (centralized)
AVAILABLE_VOICES = [
    "ash", "alloy", "ballad", "coral", "echo",
    "fable", "nova", "sage", "shimmer", "verse"
]

def _get_random_voice(exclude=None):
    import random
    if not exclude:
        pool = AVAILABLE_VOICES
    else:
        excluded = set(exclude if isinstance(exclude, (list, tuple, set)) else [exclude])
        pool = [v for v in AVAILABLE_VOICES if v not in excluded]
    if not pool:
        raise ValueError("Voice pool empty after exclusions")
    return random.choice(pool)


def _resolve_voices(conversation_type, input_voice, option_voice):
    """Return (input_voice, option_voice) based on explicit args and conversation_type.

    Rules:
    - If both voices are provided explicitly, use them as-is.
    - If only one is provided, use it for both.
    - If none provided: dialogue -> different random voices, monologue/other -> same random voice.
    """
    ct = str(conversation_type).lower() if conversation_type is not None else None

    # Explicit overrides
    if input_voice and option_voice:
        return input_voice, option_voice
    if input_voice and not option_voice:
        return input_voice, input_voice
    if option_voice and not input_voice:
        return option_voice, option_voice

    # No explicit voices -> choose by conversation type
    if ct == "dialogue":
        iv = _get_random_voice()
        ov = _get_random_voice(exclude=iv)
        return iv, ov
    else:
        iv = _get_random_voice()
        return iv, iv

