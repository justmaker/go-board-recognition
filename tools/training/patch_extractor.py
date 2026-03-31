"""
Patch Extractor — crop 32x32 patches from rendered board images.
Each patch is centered on an intersection and labeled black/white/empty.
"""

import os
import random
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from sgf_parser import BoardState, BLACK, WHITE, EMPTY
from board_renderer import IMAGE_SIZE, CELL_SIZE

PATCH_SIZE = 32
HALF = PATCH_SIZE // 2

LABEL_MAP = {BLACK: 'black', WHITE: 'white', EMPTY: 'empty'}


def _calc_margin(board_size: int) -> int:
    return (IMAGE_SIZE - (board_size - 1) * CELL_SIZE) // 2


def extract_patches(
    img: Image.Image,
    board: BoardState,
    augment: bool = False,
    seed: int = 0,
) -> list:
    """
    Extract 32x32 patches from a board image.
    Returns list of (patch_img, label_str, row, col).
    """
    rng = random.Random(seed)
    patches = []
    m = _calc_margin(board.size)

    for r in range(board.size):
        for c in range(board.size):
            cx = m + c * CELL_SIZE
            cy = m + r * CELL_SIZE

            # Crop patch
            left = cx - HALF
            top = cy - HALF
            right = cx + HALF
            bottom = cy + HALF

            patch = img.crop((left, top, right, bottom))

            if augment:
                patch = _augment_patch(patch, rng)

            label = LABEL_MAP[board.grid[r][c]]
            patches.append((patch, label, r, c))

    return patches


def _augment_patch(patch: Image.Image, rng: random.Random) -> Image.Image:
    """Apply per-patch augmentation."""
    # Small rotation ±5°
    if rng.random() < 0.5:
        angle = rng.uniform(-5, 5)
        patch = patch.rotate(angle, resample=Image.BILINEAR, fillcolor=(128, 128, 128))

    # Brightness ±20%
    if rng.random() < 0.5:
        factor = rng.uniform(0.8, 1.2)
        patch = ImageEnhance.Brightness(patch).enhance(factor)

    # Slight blur
    if rng.random() < 0.2:
        patch = patch.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.7)))

    # Gaussian noise
    if rng.random() < 0.3:
        arr = np.array(patch, dtype=np.float32)
        noise = np.random.normal(0, rng.uniform(3, 10), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        patch = Image.fromarray(arr)

    return patch


def save_patches(
    patches: list,
    output_dir: str,
    prefix: str = '',
    counter_start: int = 0,
) -> int:
    """
    Save patches to output_dir/{black,white,empty}/patch_XXXX.png
    Returns the next counter value.
    """
    for subdir in ['black', 'white', 'empty']:
        os.makedirs(os.path.join(output_dir, subdir), exist_ok=True)

    counter = counter_start
    for patch_img, label, r, c in patches:
        fname = f"{prefix}patch_{counter:06d}.png"
        path = os.path.join(output_dir, label, fname)
        patch_img.save(path)
        counter += 1

    return counter


if __name__ == '__main__':
    # Quick test
    from board_renderer import render_board

    board = BoardState(19)
    board.grid[3][3] = BLACK
    board.grid[3][15] = WHITE
    board.grid[9][9] = BLACK

    img = render_board(board, style='wood', seed=42)
    patches = extract_patches(img, board, augment=True, seed=42)
    print(f"Extracted {len(patches)} patches")

    counts = {'black': 0, 'white': 0, 'empty': 0}
    for _, label, _, _ in patches:
        counts[label] += 1
    print(f"Distribution: {counts}")

    # Save a few
    os.makedirs('output/patches', exist_ok=True)
    save_patches(patches[:10], 'output/patches', prefix='test_')
    print("Saved test patches to output/patches/")
