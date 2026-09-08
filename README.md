# 📦 PAKET LENGKAP EXPERIMENT 3 VARIAN SEGMENTASI U-NET, KLASIFIKASI & GRAD-CAM

---

## 📌 Deskripsi Paket

Folder ini (`PACKAGE_SEGMENTASI_UNET_LENGKAP`) didesain sebagai **paket mandiri 1-folder penuh** yang mencakup 3 varian arsitektur U-Net hingga pengujian interpretabilitas Grad-CAM:
1. **3 Varian Segmentasi U-Net**:
   - **ResNet18 Attention U-Net**: Model utama ber-backbone ResNet-18 ImageNet + Attention Gates.
   - **U-Net Basic**: Model U-Net standar klasik (Vanilla 4-stage Double Conv U-Net).
   - **U-Net Light**: Model U-Net ringan (16 $\rightarrow$ 32 $\rightarrow$ 64 $\rightarrow$ 128) untuk komputasi cepat.
2. **Auto-Cropping ROI Paru**: Menggunakan model utama **ResNet18 U-Net** untuk mengekstrak ROI paru-paru bagi dataset klasifikasi.
3. **Klasifikasi CNN Biner**: Training & Evaluasi model ResNet-18 + CBAM Attention pada hasil crop ResNet18 U-Net.
4. **Uji Layer Grad-CAM**: Visualisasi heatmap per-layer pada Layer 1, Layer 2, Layer 3, dan Layer 4.
5. **Uji Grad-CAM dengan Noise**: Pengujian ketahanan/robustness heatmap Grad-CAM terhadap noise Gaussian.

---

## 🛠️ 1. Instalasi Library (Dependency)

```bash
cd PACKAGE_SEGMENTASI_UNET_LENGKAP
python install_requirements.py
```
*atau menggunakan pip:*
```bash
pip install -r requirements.txt
```

---

## ⚡ 2. Opsi 1-Click Executor (Menjalankan Seluruh Pipeline Otomatis)

```bash
python JALANKAN_SEMUA_PIPELINE.py
```

Skrip ini akan secara otomatis:
1. Melatih **3 varian U-Net** (`ResNet18 Attention U-Net`, `U-Net Basic`, `U-Net Light`).
2. Auto-cropping citra ROI ke folder `DATASET/Crop_UNet_Gamma_CLAHE` menggunakan **ResNet18 U-Net**.
3. Mengevaluasi dan membandingkan metrik Dice Score, IoU, Jumlah Parameter, dan Waktu Inferensi ketiga model U-Net.
4. Melatih model klasifikasi biner ResNet-18 + CBAM pada citra hasil crop ResNet18 U-Net.
5. Mengevaluasi metrik klasifikasi (Akurasi, Precision, Recall, F1, ROC-AUC) & Confusion Matrix.
6. Menghasilkan visualisasi Grad-CAM per-layer (`Layer 1` s/d `Layer 4`).
7. Menghasilkan visualisasi & tabel uji ketahanan Grad-CAM terhadap perturbasi noise.

---

## 📂 Struktur Direktori & File

```text
PACKAGE_SEGMENTASI_UNET_LENGKAP/
│
├── 🛠️ install_requirements.py        # [INSTALL] Otomatis pasang dependency
├── 📄 requirements.txt               # [LIBRARY] PyTorch, OpenCV, Quantus, Scikit-Learn, dll
├── ⚡ JALANKAN_SEMUA_PIPELINE.py     # [1-CLICK RUN] Menjalankan seluruh eksperimen otomatis
├── 📄 README.md                      # [DOKUMENTASI] Buku panduan penggunaan
│
├── 📂 DATASET/
│   ├── 📁 Citra_Gamma_CLAHE/         # Citra input paru-paru utuh (Gamma + CLAHE) [Binary: Normal vs Malignant]
│   ├── 📁 Mask_Manual_GT/            # Mask biner segmentasi manual (Ground Truth) yang sudah di-Gamma + CLAHE
│   └── 📁 Crop_UNet_Gamma_CLAHE/     # Hasil segmentasi & crop ROI oleh ResNet18 U-Net
│
├── 📂 SCRIPTS/
│   ├── 📁 01_SEGMENTASI_UNET/
│   │   ├── modul_unet.py             # Arsitektur ResNet18 U-Net, UNet_Basic, & UNet_Light
│   │   ├── 01_train_unet.py          # Training 3 varian model U-Net
│   │   ├── 02_crop_dataset_unet.py   # Inferensi ResNet18 U-Net & auto-cropping citra
│   │   └── 03_evaluasi_unet_dice_iou.py # Evaluasi & perbandingan 3 varian U-Net
│   │
│   ├── 📁 02_KLASIFIKASI_MODEL/
│   │   ├── modul_klasifikasi.py      # Arsitektur ResNet-18 + CBAM Attention Classifier
│   │   ├── 01_train_klasifikasi.py   # Training model klasifikasi pada citra Crop ResNet18 U-Net
│   │   └── 02_evaluasi_klasifikasi.py # Evaluasi Akurasi, F1, ROC-AUC & Confusion Matrix
│   │
│   ├── 📁 03_UJI_LAYER_GRADCAM/
│   │   └── 01_uji_layer_gradcam.py   # Visualisasi heatmap Grad-CAM pada layer1, layer2, layer3, layer4
│   │
│   └── 📁 04_UJI_GRADCAM_NOISE/
│       └── 01_uji_gradcam_noise.py   # Pengujian ketahanan Grad-CAM terhadap noise Gaussian
│
└── 📂 OUTPUT_HASIL/                  # Berisi bobot model (.pt), tabel CSV, dan grafik visualisasi
```

---

## 📊 File Output yang Dihasilkan

Setelah pipeline selesai dijalankan, seluruh output akan tersimpan di dalam folder `OUTPUT_HASIL/`:

1. `best_unet_resnet18_attention_u_net.pt`, `best_unet_u_net_basic.pt`, `best_unet_u_net_light.pt` - Checkpoint bobot 3 varian U-Net.
2. `tabel_perbandingan_3_unet.csv` & `perbandingan_3_unet_metrics.png` - Tabel & grafik perbandingan Dice Score, IoU, Parameter, dan Kecepatan 3 U-Net.
3. `best_classifier.pt` - Bobot model klasifikasi ResNet-18 + CBAM terbaik.
4. `metrik_klasifikasi.csv` & `confusion_matrix_klasifikasi.png` - Evaluasi akurasi, F1, ROC-AUC, dan plot CM.
5. `perbandingan_layer_gradcam.png` - Visualisasi heatmap Grad-CAM per layer (Layer 1 s/d Layer 4).
6. `evaluasi_gradcam_noise.png` & `hasil_uji_noise.csv` - Plot visual & tabel penurunan skor korelasi Grad-CAM terhadap variasi noise.
