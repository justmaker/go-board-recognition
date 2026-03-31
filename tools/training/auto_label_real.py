"""
Auto-label real images using existing CV pipeline output.

Strategy:
1. Use the existing board_recognition pipeline to find board + grid
2. Crop 32x32 patches from real images at each intersection
3. Use the synthetic-trained CNN for initial labels
4. Output patches + labels for manual review / fine-tuning

This bridges the domain gap by adding real-world patches to training data.
"""

import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

from train import StoneCNN


def load_model(model_path: str):
    model = StoneCNN(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
    model.eval()
    return model


def predict_patch(model, patch: Image.Image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    tensor = transform(patch).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]
        pred = output.argmax(1).item()
    classes = ['black', 'empty', 'white']
    return classes[pred], {c: float(probs[i]) for i, c in enumerate(classes)}


def extract_grid_patches(image_path: str, board_size: int = 19, margin_pct: float = 0.05):
    """
    Simple grid-based extraction: assume the image is a board screenshot.
    Detect board region by looking for the dominant rectangular area,
    then divide into grid and crop patches.
    """
    img = Image.open(image_path).convert('RGB')
    w, h = img.size

    # For phone screenshots, the board is usually in the center
    # with some UI above/below. Try to find the square board region.
    # Heuristic: board occupies the width, centered vertically
    board_size_px = min(w, h)

    # If portrait (phone screenshot), board is likely width-constrained
    if h > w:
        board_left = 0
        board_top = (h - w) // 2
        board_size_px = w
    else:
        board_left = (w - h) // 2
        board_top = 0
        board_size_px = h

    # Crop the board region
    board_img = img.crop((
        board_left, board_top,
        board_left + board_size_px,
        board_top + board_size_px
    ))

    # Now extract patches at each intersection
    cell = board_size_px / board_size
    patch_size = int(cell * 0.85)
    half = patch_size // 2

    patches = []
    for r in range(board_size):
        for c in range(board_size):
            cx = int((c + 0.5) * cell)
            cy = int((r + 0.5) * cell)

            left = max(0, cx - half)
            top = max(0, cy - half)
            right = min(board_size_px, cx + half)
            bottom = min(board_size_px, cy + half)

            patch = board_img.crop((left, top, right, bottom)).resize((32, 32))
            patches.append((patch, r, c))

    return patches, board_img


def process_image(model, image_path: str, output_dir: str, board_size: int = 19):
    """Process one image: extract patches, predict labels, save."""
    basename = os.path.splitext(os.path.basename(image_path))[0]
    patches, board_img = extract_grid_patches(image_path, board_size)

    # Save cropped board for reference
    board_img.save(os.path.join(output_dir, f'{basename}_board.png'))

    labels = []
    for patch, r, c in patches:
        pred, probs = predict_patch(model, patch)
        confidence = max(probs.values())

        # Save patch
        label_dir = os.path.join(output_dir, 'patches', pred)
        os.makedirs(label_dir, exist_ok=True)
        fname = f'{basename}_r{r:02d}c{c:02d}.png'
        patch.save(os.path.join(label_dir, fname))

        labels.append({
            'file': fname,
            'row': r, 'col': c,
            'predicted': pred,
            'confidence': round(confidence, 4),
            'probs': {k: round(v, 4) for k, v in probs.items()},
        })

    # Save labels JSON for review
    labels_path = os.path.join(output_dir, f'{basename}_labels.json')
    with open(labels_path, 'w') as f:
        json.dump(labels, f, indent=2)

    # Print summary
    counts = {'black': 0, 'white': 0, 'empty': 0}
    low_conf = 0
    for l in labels:
        counts[l['predicted']] += 1
        if l['confidence'] < 0.9:
            low_conf += 1

    print(f"  {basename}: B={counts['black']} W={counts['white']} E={counts['empty']} "
          f"(low confidence: {low_conf})")

    return labels


def main():
    model_path = 'output/model/stone_classifier.pth'
    if not os.path.exists(model_path):
        print("Model not found. Run train.py first.")
        sys.exit(1)

    model = load_model(model_path)
    print(f"Loaded model: {model_path}")

    # Process test images from repo
    test_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'test_images')
    output_dir = 'output/real_patches'
    os.makedirs(output_dir, exist_ok=True)

    if not os.path.exists(test_dir):
        print(f"Test images not found at {test_dir}")
        sys.exit(1)

    all_labels = {}
    for fname in sorted(os.listdir(test_dir)):
        if not fname.lower().endswith(('.jpg', '.png')):
            continue
        fpath = os.path.join(test_dir, fname)
        print(f"Processing: {fname}")
        labels = process_image(model, fpath, output_dir, board_size=19)
        all_labels[fname] = labels

    # Summary
    print(f"\nProcessed {len(all_labels)} images")
    print(f"Patches saved to {output_dir}/patches/")
    print(f"Labels saved as *_labels.json for review")
    print(f"\nTo improve accuracy: review low-confidence labels,")
    print(f"move misclassified patches to correct folders,")
    print(f"then add to training set and retrain.")


if __name__ == '__main__':
    main()
