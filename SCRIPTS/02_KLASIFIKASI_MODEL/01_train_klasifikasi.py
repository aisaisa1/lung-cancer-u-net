import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from modul_klasifikasi import (ResNet18_CBAM, ClassificationDataset, load_dataset_files,
                               DEVICE, PACKAGE_ROOT)

EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-4
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

def train_classifier():
    print("=" * 75)
    print("🧠 TRAINING MODEL KLASIFIKASI (RESNET-18 + CBAM ATTENTION)")
    print("=" * 75)
    
    X, y = load_dataset_files('Crop_UNet_Gamma_CLAHE')
    if len(X) == 0:
        print("❌ Dataset klasifikasi tidak ditemukan!")
        return

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED, stratify=y_temp)

    print(f"📊 Split Dataset → Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(ClassificationDataset(X_train, y_train, train_tf), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(ClassificationDataset(X_val, y_val, val_tf), batch_size=BATCH_SIZE, shuffle=False)

    model = ResNet18_CBAM(num_classes=2, pretrained=True).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    save_dir = os.path.join(PACKAGE_ROOT, 'OUTPUT_HASIL')
    os.makedirs(save_dir, exist_ok=True)

    best_val_acc = 0.0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in tqdm(train_loader, desc=f"Epoch [{epoch+1:02d}/{EPOCHS}]"):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

        scheduler.step()
        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * imgs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_loss /= val_total
        val_acc = val_correct / val_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc * 100)
        history['val_acc'].append(val_acc * 100)

        print(f"📊 Epoch {epoch+1:02d} | T-Loss: {train_loss:.4f} T-Acc: {train_acc*100:.2f}% | V-Loss: {val_loss:.4f} V-Acc: {val_acc*100:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_dir, 'best_classifier.pt'))
            print(f"   ⭐️ Model Klasifikasi Terbaik Disimpan! (Val Acc: {best_val_acc*100:.2f}%)")

    np.save(os.path.join(save_dir, 'test_split_classifier.npy'), np.array(list(zip(X_test, y_test)), dtype=object))
    
    with open(os.path.join(save_dir, 'training_history_classifier.json'), 'w') as f:
        json.dump(history, f, indent=4)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Kurva Loss Klasifikasi')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Kurva Akurasi Klasifikasi')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'kurva_training_klasifikasi.png'), dpi=150)
    plt.close()

    print("=" * 75)
    print(f"🎉 TRAINING KLASIFIKASI SELESAI! Best Val Acc: {best_val_acc*100:.2f}%")
    print("=" * 75)

if __name__ == "__main__":
    train_classifier()
