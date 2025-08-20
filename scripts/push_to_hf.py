import argparse
import os
import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def path_parts(p: str):
    """Return (category, sample, filename) using the first 'data' segment as anchor.

    Expected: .../data/<category>/<sample>/<file>
    Falls back to ('misc', <parent_name>, <filename>) if not matched.
    """
    pp = Path(p)
    try:
        idx = pp.parts.index("data")
        category = pp.parts[idx + 1]
        sample = pp.parts[idx + 2]
    except Exception:
        category = "misc"
        sample = pp.parent.name
    return category, sample, pp.name


def stage_dataset(table_csv: Path, stage_dir: Path) -> Path:
    """Copy audio into stage_dir/audio/<category>/<sample>/ and write data.csv with relative paths.

    Assumes a single AUDIO column in the table.
    Returns path to staged CSV.
    """
    audio_root = stage_dir / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(table_csv)

    rel_audio = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Staging audio"):
        # AUDIO (single combined input per row)
        cat_a, samp_a, name_a = path_parts(row["AUDIO"])
        dst_dir_a = audio_root / cat_a / samp_a
        dst_dir_a.mkdir(parents=True, exist_ok=True)
        dst_path_a = dst_dir_a / name_a
        if not dst_path_a.exists():
            shutil.copy2(row["AUDIO"], dst_path_a)
        rel_audio.append(dst_path_a.relative_to(stage_dir).as_posix())

    staged = df.copy()
    staged["AUDIO"] = rel_audio
    staged_csv = stage_dir / "data.csv"
    staged.to_csv(staged_csv, index=False)
    return staged_csv


def write_readme(stage_dir: Path):
    readme = """
# Audio Consistency Checks Dataset

This dataset contains audio-based ambiguity-resolution tasks across prosody categories.

Columns:
- ROW_ID: unique row id
- GROUP_ID: <category>_<sample>
- CATEGORY: category name
- AUDIO: relative path to the single combined input audio per row. Order: target separator + target, then spoken separators and three completion audios in randomized A/B/C order.
- CORRECT_COMPLETION: one of "Completion A", "Completion B", "Completion C"
- INPUT: text of the input (target) sentence
- COMPLETION_A, COMPLETION_B, COMPLETION_C: text of the completion options
- SETTING, CONVERSATION_TYPE, VOICE_INPUT, VOICE_COMPLETION: metadata
"""
    (stage_dir / "README.md").write_text(readme.strip() + "\n")


def push_folder_to_hub(stage_dir: Path, repo_id: str, private: bool):
    from huggingface_hub import create_repo, upload_folder

    create_repo(repo_id, repo_type="dataset", private=private, exist_ok=True)
    upload_folder(folder_path=str(stage_dir), repo_id=repo_id, repo_type="dataset")


def push_processed_dataset(stage_csv: Path, repo_id: str):
    from datasets import load_dataset, Audio

    raw = load_dataset("csv", data_files=str(stage_csv))
    train = raw["train"]
    train = train.cast_column("AUDIO", Audio())
    train.push_to_hub(repo_id)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, required=True, help="<user>/<dataset> on Hugging Face Hub")
    parser.add_argument("--data_table", type=str, default="data/dataset_table.csv")
    parser.add_argument("--stage_dir", type=str, default="hf_dataset")
    parser.add_argument("--private", action="store_true", help="Create/keep the dataset repo private")
    parser.add_argument("--push_processed", action="store_true", help="Also push processed Dataset with Audio columns")

    args = parser.parse_args()
    table_csv = Path(args.data_table)
    stage_dir = Path(args.stage_dir)

    stage_dir.mkdir(parents=True, exist_ok=True)
    staged_csv = stage_dataset(table_csv, stage_dir)
    write_readme(stage_dir)
    push_folder_to_hub(stage_dir, args.repo_id, private=args.private)

    if args.push_processed:
        push_processed_dataset(staged_csv, args.repo_id)

    print("✅ Done. Repo:", args.repo_id)


if __name__ == "__main__":
    main()


