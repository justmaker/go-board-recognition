"""
Board Renderer — render board states into images with various styles.
Output: 608x608 images (19 intersections * 32px spacing).
"""

import random
import math
from PIL import Image, ImageDraw, ImageFilter, ImageEnhance
import numpy as np
from sgf_parser import BoardState, BLACK, WHITE, EMPTY

# Always render 608x608 with 32px spacing; margin varies by board size
IMAGE_SIZE = 608
CELL_SIZE = 32
MARGIN = 16  # Default margin for 19x19


def _margin(board_size: int) -> int:
    """Margin so the grid is centered in IMAGE_SIZE x IMAGE_SIZE."""
    return (IMAGE_SIZE - (board_size - 1) * CELL_SIZE) // 2


def _board_pixel_size(board_size: int) -> int:
    return (board_size - 1) * CELL_SIZE + 2 * _margin(board_size)


def _intersection_xy(row: int, col: int, board_size: int) -> tuple:
    """Get pixel coordinates of an intersection."""
    m = _margin(board_size)
    x = m + col * CELL_SIZE
    y = m + row * CELL_SIZE
    return x, y


# ============================================================
# Style: Wooden board (warm wood color + 3D stones)
# ============================================================

def _render_wood(board: BoardState, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    m = _margin(board.size)

    # Wood base color with slight random variation
    base_r = rng.randint(190, 220)
    base_g = rng.randint(160, 185)
    base_b = rng.randint(100, 130)
    img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (base_r, base_g, base_b))
    draw = ImageDraw.Draw(img)

    # Add subtle wood grain noise
    pixels = np.array(img, dtype=np.float32)
    noise = np.random.RandomState(seed).normal(0, 6, pixels.shape)
    for y_idx in range(pixels.shape[0]):
        streak = math.sin(y_idx * 0.15 + seed) * 8
        pixels[y_idx, :, :] += streak
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    draw = ImageDraw.Draw(img)

    # Draw grid lines
    line_color = (40, 30, 20)
    grid_end = m + (board.size - 1) * CELL_SIZE
    for i in range(board.size):
        pos = m + i * CELL_SIZE
        draw.line([(pos, m), (pos, grid_end)], fill=line_color, width=1)
        draw.line([(m, pos), (grid_end, pos)], fill=line_color, width=1)

    # Star points
    for r, c in _get_star_points(board.size):
        x, y = _intersection_xy(r, c, board.size)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=line_color)

    # Draw stones with 3D effect
    stone_r = int(CELL_SIZE * 0.42)
    for r in range(board.size):
        for c in range(board.size):
            color = board.grid[r][c]
            if color == EMPTY:
                continue
            x, y = _intersection_xy(r, c, board.size)
            if color == BLACK:
                for i in range(stone_r, 0, -1):
                    shade = int(20 + 30 * (1 - i / stone_r))
                    draw.ellipse([x - i, y - i, x + i, y + i],
                                fill=(shade, shade, shade))
                hx, hy = x - stone_r // 3, y - stone_r // 3
                draw.ellipse([hx - 2, hy - 2, hx + 2, hy + 2],
                            fill=(80, 80, 80))
            else:
                for i in range(stone_r, 0, -1):
                    shade = int(245 - 30 * (1 - i / stone_r))
                    draw.ellipse([x - i, y - i, x + i, y + i],
                                fill=(shade, shade, shade))
                draw.arc([x - stone_r, y - stone_r, x + stone_r, y + stone_r],
                        0, 360, fill=(180, 180, 180), width=1)

    return img


# ============================================================
# Style: App screenshot (flat, light yellow board)
# ============================================================

def _render_app(board: BoardState, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    m = _margin(board.size)

    # Light yellow/beige background
    bg_r = rng.randint(235, 250)
    bg_g = rng.randint(210, 230)
    bg_b = rng.randint(150, 175)
    img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)

    # Grid lines
    line_color = (60, 50, 40)
    grid_end = m + (board.size - 1) * CELL_SIZE
    for i in range(board.size):
        pos = m + i * CELL_SIZE
        draw.line([(pos, m), (pos, grid_end)], fill=line_color, width=1)
        draw.line([(m, pos), (grid_end, pos)], fill=line_color, width=1)

    # Star points
    for r, c in _get_star_points(board.size):
        x, y = _intersection_xy(r, c, board.size)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=line_color)

    # Flat stones
    stone_r = int(CELL_SIZE * 0.43)
    for r in range(board.size):
        for c in range(board.size):
            color = board.grid[r][c]
            if color == EMPTY:
                continue
            x, y = _intersection_xy(r, c, board.size)
            if color == BLACK:
                draw.ellipse([x - stone_r, y - stone_r, x + stone_r, y + stone_r],
                            fill=(30, 30, 30))
            else:
                draw.ellipse([x - stone_r, y - stone_r, x + stone_r, y + stone_r],
                            fill=(240, 240, 240), outline=(160, 160, 160), width=1)

    return img


# ============================================================
# Style: Dark theme (dark gray board + bright grid)
# ============================================================

def _render_dark(board: BoardState, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    m = _margin(board.size)

    bg = rng.randint(35, 55)
    img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), (bg, bg, bg + 5))
    draw = ImageDraw.Draw(img)

    # Grid lines (bright)
    line_color = (130, 130, 120)
    grid_end = m + (board.size - 1) * CELL_SIZE
    for i in range(board.size):
        pos = m + i * CELL_SIZE
        draw.line([(pos, m), (pos, grid_end)], fill=line_color, width=1)
        draw.line([(m, pos), (grid_end, pos)], fill=line_color, width=1)

    # Star points
    for r, c in _get_star_points(board.size):
        x, y = _intersection_xy(r, c, board.size)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=line_color)

    # Stones
    stone_r = int(CELL_SIZE * 0.43)
    for r in range(board.size):
        for c in range(board.size):
            color = board.grid[r][c]
            if color == EMPTY:
                continue
            x, y = _intersection_xy(r, c, board.size)
            if color == BLACK:
                draw.ellipse([x - stone_r, y - stone_r, x + stone_r, y + stone_r],
                            fill=(15, 15, 15), outline=(50, 50, 50), width=1)
            else:
                draw.ellipse([x - stone_r, y - stone_r, x + stone_r, y + stone_r],
                            fill=(230, 230, 225), outline=(180, 180, 175), width=1)

    return img


# ============================================================
# Style: 101weiqi / Fox Weiqi — green-tinted board
# ============================================================

def _render_green(board: BoardState, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    size = _board_pixel_size(board.size)
    m = _margin(board.size)

    bg_r = rng.randint(180, 200)
    bg_g = rng.randint(195, 215)
    bg_b = rng.randint(140, 165)
    img = Image.new('RGB', (size, size), (bg_r, bg_g, bg_b))
    draw = ImageDraw.Draw(img)

    line_color = (80, 70, 50)
    for i in range(board.size):
        x = MARGIN + i * CELL_SIZE
        y = MARGIN + i * CELL_SIZE
        draw.line([(x, MARGIN), (x, MARGIN + (board.size - 1) * CELL_SIZE)],
                  fill=line_color, width=1)
        draw.line([(MARGIN, y), (MARGIN + (board.size - 1) * CELL_SIZE, y)],
                  fill=line_color, width=1)

    for r, c in _get_star_points(board.size):
        x, y = _intersection_xy(r, c, board.size)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=line_color)

    stone_r = int(CELL_SIZE * 0.44)
    for r in range(board.size):
        for c in range(board.size):
            color = board.grid[r][c]
            if color == EMPTY:
                continue
            x, y = _intersection_xy(r, c, board.size)
            if color == BLACK:
                draw.ellipse([x - stone_r, y - stone_r, x + stone_r, y + stone_r],
                            fill=(20, 20, 20))
            else:
                draw.ellipse([x - stone_r, y - stone_r, x + stone_r, y + stone_r],
                            fill=(248, 248, 245), outline=(190, 190, 185), width=1)
    return img


# ============================================================
# Style: OGS-like — warm brown board with subtle texture
# ============================================================

def _render_ogs(board: BoardState, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    size = _board_pixel_size(board.size)
    m = _margin(board.size)

    base_r = rng.randint(210, 230)
    base_g = rng.randint(180, 195)
    base_b = rng.randint(130, 150)
    img = Image.new('RGB', (size, size), (base_r, base_g, base_b))
    draw = ImageDraw.Draw(img)

    line_color = (50, 40, 30)
    line_w = 1
    for i in range(board.size):
        x = MARGIN + i * CELL_SIZE
        y = MARGIN + i * CELL_SIZE
        draw.line([(x, MARGIN), (x, MARGIN + (board.size - 1) * CELL_SIZE)],
                  fill=line_color, width=line_w)
        draw.line([(MARGIN, y), (MARGIN + (board.size - 1) * CELL_SIZE, y)],
                  fill=line_color, width=line_w)

    for r, c in _get_star_points(board.size):
        x, y = _intersection_xy(r, c, board.size)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=line_color)

    stone_r = int(CELL_SIZE * 0.42)
    for r in range(board.size):
        for c in range(board.size):
            color = board.grid[r][c]
            if color == EMPTY:
                continue
            x, y = _intersection_xy(r, c, board.size)
            if color == BLACK:
                # Slight gradient
                for i_r in range(stone_r, 0, -1):
                    shade = int(15 + 25 * (1 - i_r / stone_r))
                    draw.ellipse([x - i_r, y - i_r, x + i_r, y + i_r],
                                fill=(shade, shade, shade))
            else:
                for i_r in range(stone_r, 0, -1):
                    shade = int(250 - 25 * (1 - i_r / stone_r))
                    draw.ellipse([x - i_r, y - i_r, x + i_r, y + i_r],
                                fill=(shade, shade, shade - 3))
                draw.arc([x - stone_r, y - stone_r, x + stone_r, y + stone_r],
                        0, 360, fill=(195, 195, 190), width=1)
    return img


# ============================================================
# Style: Yicheng / Tygem — reddish-brown wood
# ============================================================

def _render_redwood(board: BoardState, seed: int = 0) -> Image.Image:
    rng = random.Random(seed)
    size = _board_pixel_size(board.size)
    m = _margin(board.size)

    base_r = rng.randint(195, 215)
    base_g = rng.randint(145, 165)
    base_b = rng.randint(90, 115)
    img = Image.new('RGB', (size, size), (base_r, base_g, base_b))
    draw = ImageDraw.Draw(img)

    # Wood grain
    pixels = np.array(img, dtype=np.float32)
    for y_idx in range(pixels.shape[0]):
        streak = math.sin(y_idx * 0.12 + seed) * 6
        pixels[y_idx, :, :] += streak
    noise = np.random.RandomState(seed).normal(0, 4, pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(pixels)
    draw = ImageDraw.Draw(img)

    line_color = (35, 25, 15)
    for i in range(board.size):
        x = MARGIN + i * CELL_SIZE
        y = MARGIN + i * CELL_SIZE
        draw.line([(x, MARGIN), (x, MARGIN + (board.size - 1) * CELL_SIZE)],
                  fill=line_color, width=1)
        draw.line([(MARGIN, y), (MARGIN + (board.size - 1) * CELL_SIZE, y)],
                  fill=line_color, width=1)

    for r, c in _get_star_points(board.size):
        x, y = _intersection_xy(r, c, board.size)
        draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=line_color)

    stone_r = int(CELL_SIZE * 0.43)
    for r in range(board.size):
        for c in range(board.size):
            color = board.grid[r][c]
            if color == EMPTY:
                continue
            x, y = _intersection_xy(r, c, board.size)
            if color == BLACK:
                draw.ellipse([x - stone_r, y - stone_r, x + stone_r, y + stone_r],
                            fill=(25, 25, 25))
                hx, hy = x - stone_r // 3, y - stone_r // 3
                draw.ellipse([hx - 1, hy - 1, hx + 1, hy + 1], fill=(65, 65, 65))
            else:
                draw.ellipse([x - stone_r, y - stone_r, x + stone_r, y + stone_r],
                            fill=(242, 240, 235), outline=(175, 170, 165), width=1)
    return img


# ============================================================
# Style: High contrast screenshot (like phone screenshots)
# ============================================================

def _render_screenshot(board: BoardState, seed: int = 0) -> Image.Image:
    """Simulates a typical phone app screenshot with very clean colors."""
    rng = random.Random(seed)
    size = _board_pixel_size(board.size)
    m = _margin(board.size)

    # Very consistent, clean background — typical of app screenshots
    bg_val = rng.randint(220, 240)
    bg_g_off = rng.randint(-5, 10)
    bg_b_off = rng.randint(-30, -10)
    img = Image.new('RGB', (size, size), (bg_val, bg_val + bg_g_off, bg_val + bg_b_off))
    draw = ImageDraw.Draw(img)

    line_color = (30, 30, 30)
    for i in range(board.size):
        x = MARGIN + i * CELL_SIZE
        y = MARGIN + i * CELL_SIZE
        draw.line([(x, MARGIN), (x, MARGIN + (board.size - 1) * CELL_SIZE)],
                  fill=line_color, width=1)
        draw.line([(MARGIN, y), (MARGIN + (board.size - 1) * CELL_SIZE, y)],
                  fill=line_color, width=1)

    for r, c in _get_star_points(board.size):
        x, y = _intersection_xy(r, c, board.size)
        draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=line_color)

    stone_r = int(CELL_SIZE * 0.45)
    for r in range(board.size):
        for c in range(board.size):
            color = board.grid[r][c]
            if color == EMPTY:
                continue
            x, y = _intersection_xy(r, c, board.size)
            if color == BLACK:
                # Pure black, very clean
                draw.ellipse([x - stone_r, y - stone_r, x + stone_r, y + stone_r],
                            fill=(10, 10, 10))
            else:
                # White close to background — the hard case!
                w_val = rng.randint(245, 255)
                draw.ellipse([x - stone_r, y - stone_r, x + stone_r, y + stone_r],
                            fill=(w_val, w_val, w_val), outline=(170, 170, 165), width=1)
    return img


# ============================================================
# Utility
# ============================================================

def _get_star_points(size: int) -> list:
    """Standard star point positions."""
    if size == 19:
        pts = [3, 9, 15]
    elif size == 13:
        pts = [3, 6, 9]
    elif size == 9:
        pts = [2, 4, 6]
    else:
        return []
    return [(r, c) for r in pts for c in pts]


STYLES = {
    'wood': _render_wood,
    'app': _render_app,
    'dark': _render_dark,
    'green': _render_green,
    'ogs': _render_ogs,
    'redwood': _render_redwood,
    'screenshot': _render_screenshot,
}


def render_board(board: BoardState, style: str = 'wood', seed: int = 0) -> Image.Image:
    """Render a board state as an image."""
    renderer = STYLES.get(style, _render_wood)
    return renderer(board, seed)


def augment_image(img: Image.Image, seed: int = 0) -> Image.Image:
    """Apply random augmentations to a board image."""
    rng = random.Random(seed)

    # Brightness ±20%
    factor = rng.uniform(0.8, 1.2)
    img = ImageEnhance.Brightness(img).enhance(factor)

    # Contrast ±15%
    factor = rng.uniform(0.85, 1.15)
    img = ImageEnhance.Contrast(img).enhance(factor)

    # Slight color shift
    factor = rng.uniform(0.9, 1.1)
    img = ImageEnhance.Color(img).enhance(factor)

    # Slight blur (simulate camera)
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.3, 0.8)))

    # Gaussian noise
    if rng.random() < 0.4:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.RandomState(seed).normal(0, rng.uniform(2, 8), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(arr)

    return img


if __name__ == '__main__':
    import os
    os.makedirs('output', exist_ok=True)

    board = BoardState(19)
    board.grid[3][3] = BLACK
    board.grid[3][15] = WHITE
    board.grid[15][3] = WHITE
    board.grid[15][15] = BLACK
    board.grid[9][9] = BLACK

    for style in STYLES:
        img = render_board(board, style=style, seed=42)
        img.save(f'output/test_{style}.png')
        print(f"Saved output/test_{style}.png ({img.size})")
