import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# 1. Simple CNN Architecture for 3-class classification
# -------------------------------------------------------------
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)  # 128x128 -> 64x64
        
        # Block 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)  # 64x64 -> 32x32
        
        # Block 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)  # 32x32 -> 16x16
        
        # Block 4
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, 2)  # 16x16 -> 8x8
        
        # Fully connected classifier
        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.pool4(self.relu4(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


# -------------------------------------------------------------
# 2. Main Training Routine
# -------------------------------------------------------------
def train_model(train_dir='train', val_dir='val', epochs=5, batch_size=64, lr=0.001):
    # Select Device (Apple Silicon GPU 'mps' -> CUDA -> CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA GPU (CUDA)")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    # Data Transforms: Resizing to 128x128 for efficient training
    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # Datasets and Loaders
    print(f"Loading datasets from '{train_dir}' and '{val_dir}'...")
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    class_names = train_dataset.classes
    print(f"Detected Classes ({len(class_names)}): {class_names}")
    print(f"Train images: {len(train_dataset)} | Validation images: {len(val_dataset)}\n")

    # Initialize Model, Loss Function, and Optimizer
    model = SimpleCNN(num_classes=len(class_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }

    best_val_acc = 0.0
    start_total_time = time.time()

    print(f"Starting training for {epochs} epochs...")
    print("-" * 65)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --- Training Phase ---
        model.train()
        running_train_loss = 0.0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        epoch_train_loss = running_train_loss / train_total
        epoch_train_acc = (train_correct / train_total) * 100.0

        # --- Validation Phase ---
        model.eval()
        running_val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_val_loss += loss.item() * images.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = running_val_loss / val_total
        epoch_val_acc = (val_correct / val_total) * 100.0
        epoch_time = time.time() - epoch_start

        # Record metrics
        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)

        print(f"Epoch [{epoch}/{epochs}] ({epoch_time:.1f}s) - "
              f"Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.2f}% | "
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.2f}%")

        # Save best model checkpoint
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'classes': class_names,
                'val_acc': best_val_acc,
                'epoch': epoch
            }, 'best_model.pth')
            print(f"  -> Best model saved with Val Acc: {best_val_acc:.2f}%")

    total_time = time.time() - start_total_time
    print("-" * 65)
    print(f"Training completed in {total_time/60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")

    # ---------------------------------------------------------
    # 3. Plot & Save Training Curves
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 5))

    # Loss Curve
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), history['train_loss'], marker='o', label='Train Loss')
    plt.plot(range(1, epochs + 1), history['val_loss'], marker='o', label='Val Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    # Accuracy Curve
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), history['train_acc'], marker='o', label='Train Accuracy')
    plt.plot(range(1, epochs + 1), history['val_acc'], marker='o', label='Val Accuracy')
    plt.title('Accuracy vs Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300)
    plt.close()
    print("Training curves saved as 'training_curves.png'")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train CNN on Cat, Dog, and Wild animal images")
    parser.add_argument('--train_dir', type=str, default='train', help='Path to training directory')
    parser.add_argument('--val_dir', type=str, default='val', help='Path to validation directory')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs (default: 5)')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate (default: 0.001)')

    args = parser.parse_args()
    train_model(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr
    )

