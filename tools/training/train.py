"""
Train a small CNN stone classifier: black / white / empty.
Uses PyTorch with a lightweight custom CNN (< 1MB).

Usage:
    python train.py --data-dir output/dataset --epochs 20 --batch-size 64
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class StoneCNN(nn.Module):
    """
    Tiny CNN for 32x32 RGB patches → 3 classes.
    ~50K parameters, < 200KB when saved.
    """
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            # 32x32x3 → 16x16x16
            nn.Conv2d(3, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 16x16x16 → 8x8x32
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 8x8x32 → 4x4x64
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # 4x4x64 → 2x2x64
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    per_class_correct = [0, 0, 0]
    per_class_total = [0, 0, 0]
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)
            for i in range(3):
                mask = labels == i
                per_class_correct[i] += predicted[mask].eq(labels[mask]).sum().item()
                per_class_total[i] += mask.sum().item()
    per_class_acc = []
    for i in range(3):
        if per_class_total[i] > 0:
            per_class_acc.append(per_class_correct[i] / per_class_total[i])
        else:
            per_class_acc.append(0.0)
    return total_loss / total, correct / total, per_class_acc


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Datasets
    train_dir = os.path.join(args.data_dir, 'train')
    val_dir = os.path.join(args.data_dir, 'val')
    test_dir = os.path.join(args.data_dir, 'test')

    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(train=False))
    test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(train=False))

    print(f"Classes: {train_dataset.classes}")
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = StoneCNN(num_classes=3).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    # Training loop
    best_val_acc = 0
    os.makedirs(args.output, exist_ok=True)
    model_path = os.path.join(args.output, 'stone_classifier.pth')

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_per_class = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        lr = optimizer.param_groups[0]['lr']
        class_names = train_dataset.classes
        per_class_str = ' | '.join(f'{class_names[i]}={val_per_class[i]:.3f}' for i in range(3))

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Val loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"{per_class_str} | lr={lr:.6f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), model_path)
            print(f"  → Saved best model (val_acc={val_acc:.4f})")

    # Final evaluation on test set
    print(f"\n{'='*60}")
    print("Loading best model for test evaluation...")
    model.load_state_dict(torch.load(model_path, weights_only=True))
    test_loss, test_acc, test_per_class = evaluate(model, test_loader, criterion, device)
    class_names = train_dataset.classes
    print(f"Test accuracy: {test_acc:.4f}")
    for i, name in enumerate(class_names):
        print(f"  {name}: {test_per_class[i]:.4f}")

    # Export ONNX for potential TFLite conversion
    onnx_path = os.path.join(args.output, 'stone_classifier.onnx')
    dummy = torch.randn(1, 3, 32, 32).to(device)
    torch.onnx.export(model, dummy, onnx_path,
                      input_names=['input'], output_names=['output'],
                      dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}})
    print(f"\nExported ONNX: {onnx_path}")
    print(f"PyTorch model: {model_path}")
    print(f"Model size: {os.path.getsize(model_path) / 1024:.1f} KB")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train stone classifier CNN')
    parser.add_argument('--data-dir', default='output/dataset', help='Dataset directory')
    parser.add_argument('--output', default='output/model', help='Output directory for model')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    args = parser.parse_args()
    main(args)
