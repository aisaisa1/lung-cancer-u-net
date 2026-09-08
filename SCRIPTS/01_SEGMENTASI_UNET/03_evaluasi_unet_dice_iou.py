import os
import sys
import time
import cv2
import numpy as np
import torch
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from modul_unet import (ResNet18_UNet, UNet_Basic, UNet_Light, refine_lung_mask,
                        dice_coef, iou_score, count_parameters, DEVICE, PACKAGE_ROOT,
                        IMG_SIZE, CLASSES)

def evaluate_model_variant(model, model_name, img_base, mask_base):
    model.eval()
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    dice_list, iou_list, times_list = [], [], []

    for cls in CLASSES:
        img_dir = os.path.join(img_base, cls)
        mask_dir = os.path.join(mask_base, cls)
        if not os.path.exists(img_dir) or not os.path.exists(mask_dir):
            continue

        files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        for f in tqdm(files, desc=f"Evaluasi {model_name} [{cls}]"):
            gt_path = os.path.join(mask_dir, f)
            img_path = os.path.join(img_dir, f)
            if not os.path.exists(gt_path): continue

            img_bgr = cv2.imread(img_path)
            gt_mask = cv2.imread(gt_path, 0)
            if img_bgr is None or gt_mask is None: continue

            gt_bin = (gt_mask > 20).astype(np.uint8)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            norm = (resized.astype(np.float32) / 255.0 - mean) / std
            tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)

            t0 = time.time()
            with torch.no_grad():
                out = model(tensor)
                prob = torch.sigmoid(out).squeeze().cpu().numpy()
                pred_bin = (prob > 0.48).astype(np.uint8)
                pred_refined = refine_lung_mask(pred_bin)
            t_elapsed = (time.time() - t0) * 1000.0 # ms

            pred_full = cv2.resize(pred_refined, (gt_mask.shape[1], gt_mask.shape[0]), interpolation=cv2.INTER_NEAREST)

            d_score = dice_coef(gt_bin, pred_full)
            i_score = iou_score(gt_bin, pred_full)

            dice_list.append(d_score)
            iou_list.append(i_score)
            times_list.append(t_elapsed)

    mean_dice = np.mean(dice_list) * 100
    mean_iou = np.mean(iou_list) * 100
    avg_time = np.mean(times_list)
    params = count_parameters(model) / 1e6 # juta parameter

    return {
        'Arsitektur Model': model_name,
        'Dice Score (%)': round(mean_dice, 2),
        'IoU Score (%)': round(mean_iou, 2),
        'Jumlah Parameter (M)': round(params, 2),
        'Waktu Inferensi (ms/img)': round(avg_time, 2)
    }

def main():
    print("=" * 75)
    print("📊 EVALUASI PERBANDINGAN 3 VARIAN U-NET (RESNET18, BASIC, LIGHT)")
    print("=" * 75)

    img_base = os.path.join(PACKAGE_ROOT, 'DATASET', 'Citra_Gamma_CLAHE')
    mask_base = os.path.join(PACKAGE_ROOT, 'DATASET', 'Mask_Manual_GT')
    if not os.path.exists(img_base):
        img_base = os.path.abspath(os.path.join(PACKAGE_ROOT, '..', 'DATASET_GAMMA_CLAHE_HASIL'))
    if not os.path.exists(mask_base):
        mask_base = os.path.abspath(os.path.join(PACKAGE_ROOT, '..', 'Crop_Manual_Gamma_CLAHE'))

    output_dir = os.path.join(PACKAGE_ROOT, 'OUTPUT_HASIL')
    os.makedirs(output_dir, exist_ok=True)

    models_info = [
        ("ResNet18 Attention U-Net", ResNet18_UNet(in_ch=3, out_ch=1, pretrained=False), 'best_unet_resnet18_attention_u_net.pt'),
        ("U-Net Basic", UNet_Basic(in_channels=3, out_channels=1), 'best_unet_u_net_basic.pt'),
        ("U-Net Light", UNet_Light(in_channels=3, out_channels=1), 'best_unet_u_net_light.pt')
    ]

    comparisons = []

    for name, model_inst, ckpt_name in models_info:
        model_inst = model_inst.to(DEVICE)
        ckpt_path = os.path.join(output_dir, ckpt_name)
        if not os.path.exists(ckpt_path) and "ResNet18" in name:
            ckpt_path = os.path.join(output_dir, 'best_unet.pt')

        if os.path.exists(ckpt_path):
            model_inst.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            print(f"✅ Dimuat checkpoint: {ckpt_path}")
        else:
            print(f"⚠️ Checkpoint {ckpt_name} tidak ditemukan, menggunakan bobot inisialisasi.")

        res = evaluate_model_variant(model_inst, name, img_base, mask_base)
        comparisons.append(res)

    df_comp = pd.DataFrame(comparisons)
    csv_path = os.path.join(output_dir, 'tabel_perbandingan_3_unet.csv')
    df_comp.to_csv(csv_path, index=False)

    # Plot grafik perbandingan
    fig, ax1 = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df_comp))
    width = 0.35

    rects1 = ax1.bar(x - width/2, df_comp['Dice Score (%)'], width, label='Dice Score (%)', color='#2563eb')
    rects2 = ax1.bar(x + width/2, df_comp['IoU Score (%)'], width, label='IoU Score (%)', color='#7c3aed')

    ax1.set_ylabel('Skor Metrik (%)', fontweight='bold')
    ax1.set_title('Perbandingan Performa 3 Varian U-Net (Dice vs IoU)', fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_comp['Arsitektur Model'], fontweight='bold')
    ax1.legend()
    ax1.set_ylim(0, 105)
    ax1.grid(axis='y', linestyle='--', alpha=0.5)

    for rect in rects1 + rects2:
        height = rect.get_height()
        ax1.annotate(f'{height:.1f}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, 'perbandingan_3_unet_metrics.png')
    plt.savefig(chart_path, dpi=150)
    plt.close()

    print("\n" + "=" * 75)
    print("🏆 TABEL PERBANDINGAN PERFORMA 3 VARIAN U-NET:")
    print(df_comp.to_string(index=False))
    print(f"\n📄 Hasil CSV disimpan ke : {csv_path}")
    print(f"📊 Grafik disimpan ke     : {chart_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
