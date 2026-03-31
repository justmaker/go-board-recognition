"""
Evaluate the trained model on real-world test images from the repo.
Uses the existing board_recognition pipeline to find intersections,
then crops patches and runs CNN inference.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

# Add parent paths
sys.path.insert(0, os.path.dirname(__file__))

from train import StoneCNN


def load_model(model_path: str, device='cpu'):
    model = StoneCNN(num_classes=3)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model


def predict_patch(model, patch_img: Image.Image, device='cpu'):
    """Predict class for a single 32x32 patch."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    tensor = transform(patch_img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)
        pred = output.argmax(1).item()
    class_names = ['black', 'empty', 'white']
    return class_names[pred], probs[0].cpu().numpy()


def evaluate_real_image(model, image_path: str, board_size: int = 19):
    """
    Simple grid-based evaluation: assume the image is already a cropped board.
    Divide into grid and classify each intersection.
    """
    img = Image.open(image_path).convert('RGB')
    w, h = img.size
    
    # Assume square board image, divide into grid
    cell_w = w / board_size
    cell_h = h / board_size
    
    results = []
    for r in range(board_size):
        row = []
        for c in range(board_size):
            cx = int((c + 0.5) * cell_w)
            cy = int((r + 0.5) * cell_h)
            half = int(min(cell_w, cell_h) * 0.5)
            
            left = max(0, cx - half)
            top = max(0, cy - half)
            right = min(w, cx + half)
            bottom = min(h, cy + half)
            
            patch = img.crop((left, top, right, bottom))
            pred, probs = predict_patch(model, patch)
            row.append((pred, probs))
        results.append(row)
    
    return results


def print_board(results):
    """Print board state as text."""
    for row in results:
        line = ''
        for pred, probs in row:
            if pred == 'black':
                line += 'X '
            elif pred == 'white':
                line += 'O '
            else:
                line += '. '
        print(line)


if __name__ == '__main__':
    model_path = 'output/model/stone_classifier.pth'
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        print("Run train.py first.")
        sys.exit(1)

    model = load_model(model_path)
    print(f"Loaded model: {model_path}")

    # Test on repo test images
    test_dir = '../../test_images'
    if not os.path.exists(test_dir):
        test_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'test_images')
    
    if os.path.exists(test_dir):
        for fname in sorted(os.listdir(test_dir)):
            if fname.lower().endswith(('.jpg', '.png')):
                fpath = os.path.join(test_dir, fname)
                print(f"\n{'='*40}")
                print(f"Image: {fname}")
                print(f"{'='*40}")
                results = evaluate_real_image(model, fpath, board_size=19)
                print_board(results)
                
                # Count
                blacks = sum(1 for row in results for p, _ in row if p == 'black')
                whites = sum(1 for row in results for p, _ in row if p == 'white')
                empties = sum(1 for row in results for p, _ in row if p == 'empty')
                print(f"\nBlack={blacks}, White={whites}, Empty={empties}")
    else:
        print(f"No test images found at {test_dir}")
