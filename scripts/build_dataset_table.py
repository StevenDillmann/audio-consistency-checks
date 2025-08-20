import os
import sys
import argparse
import csv
import json
import random
import wave

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def concatenate_wavs(wav_paths, out_path, gap_ms=0):
    """Concatenate WAV files (must share audio params) into out_path.

    gap_ms: amount of silence (in milliseconds) to insert between adjacent files.
    """
    if not wav_paths:
        raise ValueError("No wav paths provided for concatenation")

    params = None
    frames = []
    silence_bytes = b""

    for p in wav_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing WAV: {p}")
        with wave.open(p, 'rb') as w:
            cur_params = (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getcomptype(), w.getcompname())
            if params is None:
                params = cur_params
                # Prepare silence bytes for requested gap
                if gap_ms and gap_ms > 0:
                    nchannels, sampwidth, framerate, _, _ = params
                    num_silent_frames = int((gap_ms / 1000.0) * framerate)
                    frame_bytes = sampwidth * nchannels
                    silence_bytes = b"\x00" * (num_silent_frames * frame_bytes)
            else:
                if cur_params != params:
                    raise ValueError(f"Incompatible WAV params: {p} has {cur_params}, expected {params}")
            frames.append(w.readframes(w.getnframes()))
            # Insert silence between files
            if silence_bytes:
                frames.append(silence_bytes)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with wave.open(out_path, 'wb') as out:
        nchannels, sampwidth, framerate, comptype, compname = params
        out.setnchannels(nchannels)
        out.setsampwidth(sampwidth)
        out.setframerate(framerate)
        out.setcomptype(comptype, compname)
        for f in frames:
            out.writeframes(f)


def build_table_for_category(category_dir, option_sep_dir, rng, overwrite=False, gap_ms=0):
    """Yield rows for CSV and create per-row stitched OPTIONS with random order.

    For each sample, we may emit up to 3 rows (one per target). For each row, we
    generate (or reuse) a dedicated stitched OPTIONS WAV with its own random
    ordering and a JSON mapping of that order. This ensures OPTIONS are random
    per-row, not shared across the sample.
    """
    # Validate separator files
    sep_map = {
        'TARGET': os.path.join(option_sep_dir, 'target.wav'),
        'A': os.path.join(option_sep_dir, 'option_a.wav'),
        'B': os.path.join(option_sep_dir, 'option_b.wav'),
        'C': os.path.join(option_sep_dir, 'option_c.wav'),
    }
    for letter, path in sep_map.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing separator WAV for {letter}: {path}")

    sample_folders = [
        os.path.join(category_dir, d)
        for d in sorted(os.listdir(category_dir))
        if d.startswith("sample_") and os.path.isdir(os.path.join(category_dir, d))
    ]

    for sample_dir in sample_folders:
        category_name = os.path.basename(category_dir)
        letters = ['A', 'B', 'C']
        idx_to_letter = ['A', 'B', 'C']

        # Load sample metadata (input_text, options_text, setting if present)
        meta_path = os.path.join(sample_dir, 'meta.json')
        meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path) as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
        input_text_meta = meta.get('input_text', '')
        options_text_meta = meta.get('options_text', [])
        setting_meta = meta.get('setting', '')
        conversation_type_meta = meta.get('conversation_type', '')
        voice_target_meta = meta.get('voice_target', '')
        voice_option_meta = meta.get('voice_option', '')
        group_id = f"{category_name}_{os.path.basename(sample_dir).replace('sample_', 'group_')}"

        # Emit up to three rows per sample, each with its own stitched order
        for i in range(3):
            audio_in = os.path.join(sample_dir, f"target{i + 1}.wav")
            if not os.path.exists(audio_in):
                continue

            combined_input_path = os.path.join(sample_dir, f'full_input{i + 1}.wav')
            order_map_path = os.path.join(sample_dir, f'options_order_target{i + 1}.json')

            if os.path.exists(order_map_path) and not overwrite:
                with open(order_map_path) as f:
                    order = json.load(f)
            else:
                order = rng.sample([0, 1, 2], 3)
                with open(order_map_path, 'w') as f:
                    json.dump(order, f)

            # Always (re)create the combined input file if overwrite or missing
            if overwrite or not os.path.exists(combined_input_path):
                input_sequence = [
                    sep_map['TARGET'],
                    audio_in,
                ] + (
                    # Reconstruct sequence to ensure combined aligns to this row's order
                    [x for pair in [(sep_map[idx_to_letter[pos]], os.path.join(sample_dir, f"option{opt_idx + 1}.wav")) for pos, opt_idx in enumerate(order)] for x in pair]
                )
                concatenate_wavs(input_sequence, combined_input_path, gap_ms=gap_ms)

            # Determine correct letter for this row's target index (i)
            try:
                pos = order.index(i)
            except ValueError:
                continue
            correct_letter = letters[pos]

            row_id = f"{group_id}_target{i + 1}"

            # Map option texts to presented letters for this row's order
            option_text_by_letter = {'A': '', 'B': '', 'C': ''}
            if isinstance(options_text_meta, list) and len(options_text_meta) == 3:
                for p, opt_idx in enumerate(order):
                    letter = letters[p]
                    if 0 <= opt_idx < 3:
                        option_text_by_letter[letter] = options_text_meta[opt_idx]

            yield {
                'ROW_ID': row_id,
                'CATEGORY': category_name,
                'GROUP_ID': group_id,
                'AUDIO': os.path.abspath(combined_input_path),
                'CORRECT_COMPLETION': f"Completion {correct_letter}",
                'INPUT': input_text_meta,
                'OPTION_A': option_text_by_letter['A'],
                'OPTION_B': option_text_by_letter['B'],
                'OPTION_C': option_text_by_letter['C'],
                'SETTING': setting_meta,
                'CONVERSATION_TYPE': conversation_type_meta,
                'VOICE_INPUT': voice_target_meta,
                'VOICE_COMPLETION': voice_option_meta,
            }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True, help='Root data directory containing category subfolders (e.g., data/)')
    parser.add_argument('--option_sep_dir', type=str, required=True, help='Directory with separator WAVs: option_a.wav, option_b.wav, option_c.wav')
    parser.add_argument('--out_csv', type=str, required=True, help='Output CSV path')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for option ordering')
    parser.add_argument('--overwrite', action='store_true', help='Rebuild stitched options even if present')
    parser.add_argument('--gap_ms', type=int, default=250, help='Silence (ms) inserted between separator and option, and between options')
    parser.add_argument('--shuffle_rows', action='store_true', help='Shuffle CSV rows before writing')
    parser.add_argument('--shuffle_seed', type=int, default=None, help='Seed for row shuffling (defaults to --seed if not set)')
    # ROW_ID format is fixed as '<GROUP_ID>_target<T_IDX>'

    args = parser.parse_args()
    rng = random.Random(args.seed)

    # Find category dirs (those containing sample_* subdirs)
    category_dirs = [
        os.path.join(args.data_root, d)
        for d in sorted(os.listdir(args.data_root))
        if os.path.isdir(os.path.join(args.data_root, d))
    ]

    rows = []
    for category_dir in category_dirs:
        # Only process if it contains at least one sample_* folder
        if not any(name.startswith('sample_') for name in os.listdir(category_dir)):
            continue
        for row in build_table_for_category(category_dir, args.option_sep_dir, rng, overwrite=args.overwrite, gap_ms=args.gap_ms):
            # Construct a deterministic ROW_ID: <GROUP_ID>_target<T_IDX>
            try:
                audio_name = os.path.basename(row.get('AUDIO', ''))
                # Expect pattern like full_input2.wav
                if 'full_input' in audio_name:
                    t_idx = int(audio_name.replace('full_input', '').split('.')[0])
                else:
                    t_idx = 0
                row['ROW_ID'] = f"{row.get('GROUP_ID', '')}_target{t_idx}"
            except Exception:
                row['ROW_ID'] = f"{row.get('GROUP_ID','')}_target1"
            rows.append(row)

    # Optional shuffle of rows
    if args.shuffle_rows:
        shuffle_seed = args.shuffle_seed if args.shuffle_seed is not None else args.seed
        random.Random(shuffle_seed).shuffle(rows)

    # Write CSV
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    with open(args.out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'ROW_ID',
                'GROUP_ID',
                'CATEGORY',
                'AUDIO',
                'CORRECT_COMPLETION',
                'INPUT',
                'OPTION_A',
                'OPTION_B',
                'OPTION_C',
                'SETTING',
                'CONVERSATION_TYPE',
                'VOICE_INPUT',
                'VOICE_COMPLETION',
            ],
            delimiter=',',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"✅ Wrote table with {len(rows)} rows to {args.out_csv}")


if __name__ == '__main__':
    main()


# USAGE:
# python scripts/build_dataset_table.py --data_root data --option_sep_dir data/option_separation --out_csv data/dataset_table.csv --seed 123 --gap_ms 250 --overwrite --shuffle_rows --shuffle_seed 123
