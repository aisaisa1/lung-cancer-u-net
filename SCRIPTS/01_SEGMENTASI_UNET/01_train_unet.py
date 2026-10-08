import os
import sys
import random
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from modul_unet import (ResNet18_UNet, UNet_Basic, UNet_Light, LungSegDataset, build_pairs,
                        HybridIoUSuperLoss, DEVICE, PACKAGE_ROOT, IMG_SIZE)

EPOCHS = 15
BATCH_SIZE = 16
LR = 2e-4
WEIGHT_DECAY = 1e-4
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

def split_pairs(pairs_by_class, val_ratio=0.15, test_ratio=0.15):
    train, val, test = [], [], []
    for cls, pairs in pairs_by_class.items():
        random.shuffle(pairs)
        n = len(pairs)
        n_val = int(n * val_ratio)
        n_test = int(n * test_ratio)
        train += pairs[: n - n_val - n_test]
        val += pairs[n - n_val - n_test: n - n_test]
        test += pairs[n - n_test:]
    return train, val, test

def calculate_metrics(preds, targets, smooth=1e-6):
    preds_flat = preds.view(-1)
    targets_flat = targets.view(-1)
    tp = (preds_flat * targets_flat).sum().item()
    fp = (preds_flat * (1.0 - targets_flat)).sum().item()
    fn = ((1.0 - preds_flat) * targets_flat).sum().item()
    dice = (2.0 * tp + smooth) / (2.0 * tp + fp + fn + smooth)
    iou = (tp + smooth) / (tp + fp + fn + smooth)
    return dice, iou

def train_single_model(model_name, model, train_loader, val_loader, save_dir):
    print("\n" + "=" * 70)
    print(f"🔥 TRAINING VARIAN MODEL: {model_name}")
    print("=" * 70)

    criterion = HybridIoUSuperLoss().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    best_val_iou = 0.0
    history = {'train_loss': [], 'val_loss': [], 'val_dice': [], 'val_iou': []}

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        pbar = tqdm(train_loader, desc=f"[{model_name}] Epoch [{epoch+1:02d}/{EPOCHS}]")
        
        for imgs, masks in pbar:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * imgs.size(0)

        train_loss /= len(train_loader.dataset)
        scheduler.step()

        model.eval()
        val_loss = 0.0
        total_dice, total_iou = 0.0, 0.0
        val_batches = 0

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
                out = model(imgs)
                val_loss += criterion(out, masks).item() * imgs.size(0)
                probs = torch.sigmoid(out)
                preds = (probs > 0.48).float()
                d, i = calculate_metrics(preds, masks)
                total_dice += d
                total_iou += i
                val_batches += 1

        val_loss /= len(val_loader.dataset)
        val_dice = (total_dice / val_batches) * 100
        val_iou = (total_iou / val_batches) * 100

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)
        history['val_iou'].append(val_iou)

        print(f"📊 {model_name} | Epoch {epoch+1:02d} | T-Loss: {train_loss:.4f} | V-Loss: {val_loss:.4f} | Dice: {val_dice:.2f}% | IoU: {val_iou:.2f}%")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_filename = f"best_unet_{model_name.lower().replace('-', '_').replace(' ', '_')}.pt"
            torch.save(model.state_dict(), os.path.join(save_dir, save_filename))
            if "ResNet18" in model_name:
                torch.save(model.state_dict(), os.path.join(save_dir, 'best_unet.pt'))
            print(f"   ⭐️ Model {model_name} Terbaik Disimpan! (Val IoU: {best_val_iou:.2f}%)")

    return history, best_val_iou

def train_all_models():
    print("=" * 75)
    print("🚀 MEMULAI TRAINING 3 VARIAN SEGMENTASI U-NET (RESNET18, BASIC, LIGHT)")
    print("=" * 75)

    pairs_by_class = build_pairs(
        img_subfolder='Citra_Gamma_CLAHE',
        package_root=PACKAGE_ROOT
    )

    train_pairs, val_pairs, test_pairs = split_pairs(pairs_by_class)
    train_ds = LungSegDataset(train_pairs, size=IMG_SIZE, augment=True)
    val_ds = LungSegDataset(val_pairs, size=IMG_SIZE, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    save_dir = os.path.join(PACKAGE_ROOT, 'OUTPUT_HASIL')
    os.makedirs(save_dir, exist_ok=True)

    models_to_train = [
        ("ResNet18 Attention U-Net", ResNet18_UNet(in_ch=3, out_ch=1, pretrained=True).to(DEVICE)),
        ("U-Net Basic", UNet_Basic(in_channels=3, out_channels=1).to(DEVICE)),
        ("U-Net Light", UNet_Light(in_channels=3, out_channels=1).to(DEVICE))
    ]

    all_histories = {}
    summary_results = []

    for name, model_inst in models_to_train:
        hist, best_iou = train_single_model(name, model_inst, train_loader, val_loader, save_dir)
        all_histories[name] = hist
        summary_results.append({'model': name, 'best_val_iou': best_iou})

    with open(os.path.join(save_dir, 'training_histories_3_unet.json'), 'w') as f:
        json.dump(all_histories, f, indent=4)

    print("\n" + "=" * 75)
    print("🎉 SELURUH 3 VARIAN MODEL U-NET SELESAI DITRAIN!")
    for res in summary_results:
        print(f"   - {res['model']}: Best Val IoU = {res['best_val_iou']:.2f}%")
    print("=" * 75)

if __name__ == "__main__":
    train_all_models()
