"""
CLI utility to prepare BigEarthNet adaptation instruction-tuning data.
Usage:
    python training/prepare_bigearthnet.py --data_dir data/bigearthnet_v2 --output data/bigearthnet_v2/train_instructions.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.datasets.bigearthnet import BigEarthNetAdapter
from training.preprocessing.prepare_instruction_data import export_instruction_dataset


def main():
    parser = argparse.ArgumentParser(description="Prepare BigEarthNet VLM instruction-tuning data")
    parser.add_argument("--data_dir", type=str, default="data/bigearthnet_v2", help="Path to BigEarthNet dataset directory")
    parser.add_argument("--annotation_file", type=str, default=None, help="Explicit path to annotations file")
    parser.add_argument("--output", type=str, default="data/bigearthnet_v2/train_instructions.jsonl", help="Output JSONL path")
    parser.add_argument("--val_split", type=float, default=0.1, help="Validation fraction")
    args = parser.parse_args()

    adapter = BigEarthNetAdapter(data_dir=args.data_dir)
    if not adapter.is_available() and args.annotation_file is None:
        print(f"[BigEarthNet Preparation] Dataset not found in '{args.data_dir}'.")
        print("[BigEarthNet Preparation] Status: NOT_DOWNLOADED. Run dataset download instructions first.")
        return

    print(f"[BigEarthNet Preparation] Reading annotations from: {args.annotation_file or args.data_dir}")
    records = adapter.parse_annotations(annotation_file=args.annotation_file)
    print(f"[BigEarthNet Preparation] Parsed {len(records)} verified image patch records.")

    if not records:
        print("[BigEarthNet Preparation] No records found to process.")
        return

    examples = adapter.generate_instruction_examples(records)
    print(f"[BigEarthNet Preparation] Generated {len(examples)} instruction-tuning examples.")

    out_path = Path(args.output)
    count = export_instruction_dataset(examples, str(out_path))
    print(f"[BigEarthNet Preparation] Successfully exported {count} samples to '{out_path}'.")


if __name__ == "__main__":
    main()
