#!/usr/bin/env python3
"""
Generate Dataset — full pipeline: SGF → render → patches → train/val/test split.

Usage:
    python generate_dataset.py --sgf-dir sgf_data/ --output output/dataset \
        --samples-per-game 10 --styles wood,app,dark --augment
"""

import argparse
import os
import random
import shutil
from pathlib import Path

from sgf_parser import replay_game, BLACK, WHITE
from board_renderer import render_board, augment_image, STYLES
from patch_extractor import extract_patches, save_patches, LABEL_MAP


def collect_sgf_files(sgf_dir: str) -> list:
    """Find all .sgf files recursively."""
    sgf_files = []
    for root, dirs, files in os.walk(sgf_dir):
        for f in files:
            if f.lower().endswith('.sgf'):
                sgf_files.append(os.path.join(root, f))
    return sorted(sgf_files)


def generate(args):
    sgf_files = collect_sgf_files(args.sgf_dir)
    if not sgf_files:
        print(f"No SGF files found in {args.sgf_dir}")
        return

    print(f"Found {len(sgf_files)} SGF files")

    styles = args.styles.split(',')
    print(f"Styles: {styles}")

    # Prepare output
    patches_dir = os.path.join(args.output, 'all_patches')
    os.makedirs(patches_dir, exist_ok=True)

    counter = 0
    total_patches = 0
    counts = {'black': 0, 'white': 0, 'empty': 0}

    for sgf_idx, sgf_path in enumerate(sgf_files):
        print(f"\n[{sgf_idx + 1}/{len(sgf_files)}] {os.path.basename(sgf_path)}")

        try:
            with open(sgf_path, 'r', encoding='utf-8', errors='ignore') as f:
                sgf_text = f.read()
        except Exception as e:
            print(f"  Error reading: {e}")
            continue

        # Replay and sample board states
        sample_every = max(1, 200 // args.samples_per_game)
        states = replay_game(sgf_text, sample_every=sample_every)
        print(f"  Sampled {len(states)} board states (every {sample_every} moves)")

        for state_idx, board in enumerate(states):
            for style in styles:
                seed = sgf_idx * 10000 + state_idx * 100 + hash(style) % 100

                # Render board
                img = render_board(board, style=style, seed=seed)

                # Optionally augment the whole image
                if args.augment:
                    img = augment_image(img, seed=seed + 1)

                # Extract patches
                patches = extract_patches(
                    img, board,
                    augment=args.augment,
                    seed=seed + 2,
                )

                # Count
                for _, label, _, _ in patches:
                    counts[label] += 1

                prefix = f"g{sgf_idx:03d}_s{state_idx:03d}_{style}_"
                counter = save_patches(patches, patches_dir, prefix=prefix,
                                       counter_start=counter)
                total_patches = counter

        print(f"  Running total: {total_patches} patches")

    print(f"\n=== Total: {total_patches} patches ===")
    print(f"Distribution: {counts}")

    # Balance classes by undersampling the majority class
    if args.balance:
        print("\nBalancing classes...")
        _balance_classes(patches_dir, counts)

    # Train/val/test split
    print(f"\nSplitting into train/val/test ({args.split})...")
    _split_dataset(patches_dir, args.output, args.split)

    print("\nDone!")


def _balance_classes(patches_dir: str, counts: dict):
    """Undersample majority classes to match the minority class count."""
    min_count = min(counts.values())
    print(f"  Target per class: {min_count}")

    for label in ['black', 'white', 'empty']:
        label_dir = os.path.join(patches_dir, label)
        if not os.path.exists(label_dir):
            continue
        files = sorted(os.listdir(label_dir))
        if len(files) <= min_count:
            continue
        # Randomly remove excess
        remove = random.sample(files, len(files) - min_count)
        for f in remove:
            os.remove(os.path.join(label_dir, f))
        print(f"  {label}: {len(files)} → {min_count}")


def _split_dataset(patches_dir: str, output_dir: str, split: str):
    """Split patches into train/val/test."""
    ratios = [float(x) for x in split.split('/')]
    assert len(ratios) == 3, "Split must be 3 values like 80/10/10"
    total = sum(ratios)
    ratios = [r / total for r in ratios]

    for split_name in ['train', 'val', 'test']:
        for label in ['black', 'white', 'empty']:
            os.makedirs(os.path.join(output_dir, split_name, label), exist_ok=True)

    for label in ['black', 'white', 'empty']:
        label_dir = os.path.join(patches_dir, label)
        if not os.path.exists(label_dir):
            continue
        files = sorted(os.listdir(label_dir))
        random.shuffle(files)

        n_train = int(len(files) * ratios[0])
        n_val = int(len(files) * ratios[1])

        splits = {
            'train': files[:n_train],
            'val': files[n_train:n_train + n_val],
            'test': files[n_train + n_val:],
        }

        for split_name, split_files in splits.items():
            dest_dir = os.path.join(output_dir, split_name, label)
            for f in split_files:
                src = os.path.join(label_dir, f)
                dst = os.path.join(dest_dir, f)
                shutil.copy2(src, dst)
            print(f"  {label}/{split_name}: {len(split_files)} patches")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate training dataset from SGF files')
    parser.add_argument('--sgf-dir', default='sgf_data', help='Directory containing SGF files')
    parser.add_argument('--output', default='output/dataset', help='Output directory')
    parser.add_argument('--samples-per-game', type=int, default=10,
                        help='Number of board states to sample per game')
    parser.add_argument('--styles', default='wood,app,dark',
                        help='Comma-separated rendering styles')
    parser.add_argument('--augment', action='store_true', help='Apply data augmentation')
    parser.add_argument('--balance', action='store_true', help='Balance class distribution')
    parser.add_argument('--split', default='80/10/10', help='Train/val/test split ratio')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    random.seed(args.seed)
    generate(args)
