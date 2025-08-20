import os
import sys
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.generate_audio import generate_audio_sample

def main(args):
    base_dir = args.category_dir
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Directory not found: {base_dir}")

    sample_folders = [
        os.path.join(base_dir, d)
        for d in sorted(os.listdir(base_dir))
        if d.startswith("sample_") and os.path.isdir(os.path.join(base_dir, d))
    ]

    if not sample_folders:
        raise FileNotFoundError(f"No sample folders found in {base_dir}")
    
    for folder in sample_folders:
        try:
            print(f"\n🔊 Generating audio for: {folder}")
            # Determine voices
            input_voice = None
            option_voice = None

            # Priority 1: explicit per-role voices
            if args.input_voice:
                input_voice = args.input_voice
            if args.option_voice:
                option_voice = args.option_voice

            # Priority 2: single voice for both
            if args.voice and (input_voice is None and option_voice is None):
                input_voice = args.voice
                option_voice = args.voice

            # Otherwise, let utils.generate_audio infer from meta.json (conversation_type) or default to monologue
            generate_audio_sample(folder, input_voice=input_voice, option_voice=option_voice)
        except Exception as e:
            print(f"❌ Failed for {folder}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--category_dir", type=str, required=True, help = "Path to the category directory")
    parser.add_argument("--voice", type=str, default=None, help="Single voice for both input and options (fallback)")
    parser.add_argument("--input_voice", type=str, default=None, help="Voice for target/input utterances")
    parser.add_argument("--option_voice", type=str, default=None, help="Voice for follow-up/option utterances")
    # No different_voices flag here; this is now governed by each sample's meta.json
    args = parser.parse_args()
    main(args)

# USAGE:
# python scripts/generate_audio_batch.py --category_dir data/lexical_stress_shift 
# With explicit different voices: 
# python scripts/generate_audio_batch.py --category_dir data/lexical_stress_shift --input_voice ash --option_voice nova
# Otherwise, voice assignment is determined per sample by meta.json (conversation_type)
