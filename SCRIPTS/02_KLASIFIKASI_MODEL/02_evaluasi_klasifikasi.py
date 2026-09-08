import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from modul_klasifikasi import (ResNet18_CBAM, ClassificationDataset, load_dataset_files,
                               DEVICE, PACKAGE_ROOT, CLASSES_BINARY)

def evaluate_classifier():
    print("=" * 75)
    print("📊 EVALUASI PERFORMA MODEL KLASIFIKASI (AKURASI, F1, ROC-AUC)")
    print("=" * 75)

    save_dir = os.path.join(PACKAGE_ROOT, 'OUTPUT_HASIL')
    ckpt_path = os.path.join(save_dir, 'best_classifier.pt')

    model = ResNet18_CBAM(num_classes=2, pretrained=False).to(DEVICE)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"✅ Loaded Classifier Model: {ckpt_path}")
    else:
        print("⚠️ Warning: Model checkpoint classifier tidak ditemukan!")

    model.eval()

    test_split_path = os.path.join(save_dir, 'test_split_classifier.npy')
    if os.path.exists(test_split_path):
        test_pairs = np.load(test_split_path, allow_pickle=True)
        X_test = [p[0] for p in test_pairs]
        y_test = [p[1] for p in test_pairs]
    else:
        X, y = load_dataset_files('Crop_UNet_Gamma_CLAHE')
        X_test, y_test = X[:100], y[:100]

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_loader = DataLoader(ClassificationDataset(X_test, y_test, val_tf), batch_size=16, shuffle=False)

    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = outputs.argmax(dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds) * 100
    prec = precision_score(all_labels, all_preds) * 100
    rec = recall_score(all_labels, all_preds) * 100
    f1 = f1_score(all_labels, all_preds) * 100
    auc = roc_auc_score(all_labels, all_probs) * 100

    cm = confusion_matrix(all_labels, all_preds)

    print("=" * 75)
    print("🏆 METRIK EVALUASI KLASIFIKASI RESNET-18 + CBAM:")
    print(f"   - Akurasi   : {acc:.2f}%")
    print(f"   - Precision : {prec:.2f}%")
    print(f"   - Recall    : {rec:.2f}%")
    print(f"   - F1-Score  : {f1:.2f}%")
    print(f"   - ROC-AUC   : {auc:.2f}%")
    print("=" * 75)

    metrics_df = pd.DataFrame([{
        'Akurasi (%)': acc,
        'Precision (%)': prec,
        'Recall (%)': rec,
        'F1-Score (%)': f1,
        'ROC-AUC (%)': auc
    }])
    metrics_df.to_csv(os.path.join(save_dir, 'metrik_klasifikasi.csv'), index=False)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASSES_BINARY, yticklabels=CLASSES_BINARY)
    plt.title('Confusion Matrix Klasifikasi U-Net Crop')
    plt.xlabel('Prediksi')
    plt.ylabel('Ground Truth')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'confusion_matrix_klasifikasi.png'), dpi=150)
    plt.close()

if __name__ == "__main__":
    evaluate_classifier()
