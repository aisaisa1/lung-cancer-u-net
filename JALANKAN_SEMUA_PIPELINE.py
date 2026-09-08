import os
import sys
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def run_script(script_path, description):
    print("\n" + "=" * 80)
    print(f"🚀 RUNNING STEP: {description}")
    print(f"📄 Path: {script_path}")
    print("=" * 80 + "\n")
    
    start_time = time.time()
    result = subprocess.run([sys.executable, script_path], cwd=SCRIPT_DIR)
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✅ {description} SELESAI DENGAN SUKSES! (Waktu: {elapsed:.2f} detik)")
    else:
        print(f"\n❌ ERROR: {description} gagal dieksekusi (Exit Code: {result.returncode})")
        sys.exit(result.returncode)

def main():
    print("=" * 80)
    print("🌟 1-CLICK PIPELINE LENGKAP: 3 VARIAN U-NET (RESNET18, BASIC, LIGHT) + KLASIFIKASI + GRAD-CAM")
    print("=" * 80)
    
    pipeline_steps = [
        (
            os.path.join(SCRIPT_DIR, "SCRIPTS", "01_SEGMENTASI_UNET", "01_train_unet.py"),
            "1. Training 3 Varian Model Segmentasi U-Net (ResNet18 Attention U-Net, U-Net Basic, U-Net Light)"
        ),
        (
            os.path.join(SCRIPT_DIR, "SCRIPTS", "01_SEGMENTASI_UNET", "02_crop_dataset_unet.py"),
            "2. Inferensi ResNet18 U-Net & Auto-Cropping Dataset (Crop_UNet_Gamma_CLAHE)"
        ),
        (
            os.path.join(SCRIPT_DIR, "SCRIPTS", "01_SEGMENTASI_UNET", "03_evaluasi_unet_dice_iou.py"),
            "3. Evaluasi Perbandingan 3 Varian U-Net (Dice Score, IoU, Parameter, Speed)"
        ),
        (
            os.path.join(SCRIPT_DIR, "SCRIPTS", "02_KLASIFIKASI_MODEL", "01_train_klasifikasi.py"),
            "4. Training Model Klasifikasi (ResNet-18 + CBAM Attention pada Hasil Crop ResNet18 U-Net)"
        ),
        (
            os.path.join(SCRIPT_DIR, "SCRIPTS", "02_KLASIFIKASI_MODEL", "02_evaluasi_klasifikasi.py"),
            "5. Evaluasi Performa Klasifikasi (Akurasi, F1-Score, ROC-AUC, Confusion Matrix)"
        ),
        (
            os.path.join(SCRIPT_DIR, "SCRIPTS", "03_UJI_LAYER_GRADCAM", "01_uji_layer_gradcam.py"),
            "6. Pengujian Heatmap Grad-CAM per Layer (Layer 1, Layer 2, Layer 3, Layer 4)"
        ),
        (
            os.path.join(SCRIPT_DIR, "SCRIPTS", "04_UJI_GRADCAM_NOISE", "01_uji_gradcam_noise.py"),
            "7. Pengujian Ketahanan Heatmap Grad-CAM terhadap Perturbasi Noise Gaussian"
        )
    ]
    
    start_total = time.time()
    for script_path, desc in pipeline_steps:
        run_script(script_path, desc)
        
    total_elapsed = time.time() - start_total
    print("\n" + "=" * 80)
    print("🎉 SELURUH PIPELINE EXPERIMENT SELESAI 100%!")
    print(f"⏱️ Total Waktu Eksekusi : {total_elapsed / 60:.2f} menit")
    print(f"📁 Seluruh hasil output tersimpan di: {os.path.join(SCRIPT_DIR, 'OUTPUT_HASIL')}")
    print("=" * 80)

if __name__ == "__main__":
    main()
