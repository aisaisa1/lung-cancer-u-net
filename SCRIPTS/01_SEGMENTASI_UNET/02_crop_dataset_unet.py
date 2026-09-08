import os
import sys
import cv2
import numpy as np
import torch
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from modul_unet import (ResNet18_UNet, UNet_Basic, UNet_Light, refine_lung_mask,
                        DEVICE, PACKAGE_ROOT, IMG_SIZE, CLASSES)

def crop_image_with_mask(img, mask):
    masked_img = cv2.bitwise_and(img, img, mask=mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    x_min, y_min = img.shape[1], img.shape[0]
    x_max, y_max = 0, 0
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > 100:
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x + w), max(y_max, y + h)
            
    if x_max > x_min and y_max > y_min:
        cropped = masked_img[y_min:y_max, x_min:x_max]
    else:
        cropped = masked_img
        
    return cv2.resize(cropped, (IMG_SIZE, IMG_SIZE))

def crop_dataset_for_model(model_name, model, ckpt_filename, output_subfolder_name):
    print("\n" + "=" * 75)
    print(f"✂️ MENJALANKAN BULK AUTO-CROP UNTUK MODEL: {model_name}")
    print("=" * 75)

    ckpt_path = os.path.join(PACKAGE_ROOT, 'OUTPUT_HASIL', ckpt_filename)
    if not os.path.exists(ckpt_path) and "ResNet18" in model_name:
        ckpt_path = os.path.join(PACKAGE_ROOT, 'OUTPUT_HASIL', 'best_unet.pt')

    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"✅ Loaded Checkpoint: {ckpt_path}")
    else:
        print(f"⚠️ Checkpoint {ckpt_filename} belum ada, menggunakan bobot inisialisasi.")

    model.eval()

    input_base = os.path.join(PACKAGE_ROOT, 'DATASET', 'Citra_Gamma_CLAHE')
    if not os.path.exists(input_base):
        input_base = os.path.abspath(os.path.join(PACKAGE_ROOT, '..', 'DATASET_GAMMA_CLAHE_HASIL'))

    output_base = os.path.join(PACKAGE_ROOT, 'DATASET', output_subfolder_name)
    os.makedirs(output_base, exist_ok=True)

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    total_processed = 0
    for cls in CLASSES:
        cls_in = os.path.join(input_base, cls)
        cls_out = os.path.join(output_base, cls)
        if not os.path.exists(cls_in):
            continue
        os.makedirs(cls_out, exist_ok=True)
        
        files = [f for f in os.listdir(cls_in) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"📂 [{model_name}] Memproses {len(files)} citra kelas '{cls}'...")
        
        for f in tqdm(files, desc=f"Cropping {model_name} [{cls}]"):
            img_path = os.path.join(cls_in, f)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None: continue
            
            orig_h, orig_w = img_bgr.shape[:2]
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
            
            norm = (resized.astype(np.float32) / 255.0 - mean) / std
            tensor = torch.from_numpy(norm.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                out = model(tensor)
                prob = torch.sigmoid(out).squeeze().cpu().numpy()
                mask_bin = (prob > 0.48).astype(np.uint8)
                mask_refined = refine_lung_mask(mask_bin)
                
            mask_orig = cv2.resize(mask_refined, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
            cropped_img = crop_image_with_mask(img_bgr, mask_orig)
            
            save_path = os.path.join(cls_out, f)
            cv2.imwrite(save_path, cropped_img)
            total_processed += 1

    print(f"🎉 [{model_name}] Cropping selesai! Total {total_processed} citra disimpan ke:")
    print(f"   📁 {output_base}")

def main():
    print("=" * 80)
    print("✂️ MENJALANKAN BULK AUTO-CROP DATASET UNTUK 3 VERSI MODEL U-NET")
    print("=" * 80)

    models_to_crop = [
        ("ResNet18 Attention U-Net", ResNet18_UNet(in_ch=3, out_ch=1, pretrained=False).to(DEVICE), 'best_unet_resnet18_attention_u_net.pt', 'Crop_UNet_Gamma_CLAHE'),
        ("U-Net Basic", UNet_Basic(in_channels=3, out_channels=1).to(DEVICE), 'best_unet_u_net_basic.pt', 'Crop_UNet_Basic_Gamma_CLAHE'),
        ("U-Net Light", UNet_Light(in_channels=3, out_channels=1).to(DEVICE), 'best_unet_u_net_light.pt', 'Crop_UNet_Light_Gamma_CLAHE')
    ]

    for name, model_inst, ckpt_file, out_folder in models_to_crop:
        crop_dataset_for_model(name, model_inst, ckpt_file, out_folder)

    print("\n" + "=" * 80)
    print("🎉 AUTO-CROP KETIGA VERSI U-NET SELESAI 100%!")
    print("=" * 80)

if __name__ == "__main__":
    main()
