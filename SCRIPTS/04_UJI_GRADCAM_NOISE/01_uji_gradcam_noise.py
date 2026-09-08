import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KLASIFIKASI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../02_KLASIFIKASI_MODEL'))
sys.path.insert(0, KLASIFIKASI_DIR)
from modul_klasifikasi import ResNet18_CBAM, DEVICE, PACKAGE_ROOT

OUTPUT_DIR = os.path.join(PACKAGE_ROOT, 'OUTPUT_HASIL')
os.makedirs(OUTPUT_DIR, exist_ok=True)

class GradCAM_Binary:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self.hook_handles.append(target_layer.register_forward_hook(self.save_activation))
        self.hook_handles.append(target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        for h in self.hook_handles: h.remove()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        output = self.model(x)
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        output[0, class_idx].backward(retain_graph=True)
        grads = self.gradients[0].cpu().numpy()
        acts = self.activations[0].cpu().numpy()
        weights = np.mean(grads, axis=(1, 2))
        
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        img_np = x[0].cpu().numpy().transpose(1, 2, 0)
        img_gray = np.mean(img_np, axis=2)
        img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)
        mask = (img_gray > 0.08).astype(np.float32)
        cam = cam * mask
        
        cam = cv2.GaussianBlur(cam, (21, 21), 0)
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, class_idx

def add_gaussian_noise(img_np, sigma):
    noise = np.random.normal(0, sigma * 255, img_np.shape)
    noisy = np.clip(img_np.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy

def main():
    print("=" * 75)
    print("⚡ UJI KETAHANAN GRAD-CAM TERHADAP NOISE & PERTURBASI")
    print("=" * 75)

    ckpt_path = os.path.join(OUTPUT_DIR, "best_classifier.pt")
    model = ResNet18_CBAM(num_classes=2, pretrained=False).to(DEVICE)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"✅ Loaded Classifier Model: {ckpt_path}")
    else:
        print("⚠️ Classifier model checkpoint tidak ditemukan!")

    model.eval()
    grad_cam = GradCAM_Binary(model, model.cbam4)

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset_crop = os.path.join(PACKAGE_ROOT, 'DATASET', 'Crop_UNet_Gamma_CLAHE')
    if not os.path.exists(dataset_crop):
        dataset_crop = os.path.abspath(os.path.join(PACKAGE_ROOT, '..', 'Crop_UNet_Gamma_CLAHE'))

    samples = [
        {"class_name": "Normal", "folder": "Normal cases", "filename": "Normal case (10).jpg", "target_cls": 0},
        {"class_name": "Cancer", "folder": "Malignant cases", "filename": "Malignant case (301).jpg", "target_cls": 1}
    ]

    sigmas = [0.0, 0.05, 0.10, 0.20, 0.30]
    results = []

    fig, axes = plt.subplots(len(samples), len(sigmas) + 1, figsize=(16, 3.2 * len(samples)), dpi=200)

    for r_idx, sample in enumerate(samples):
        img_path = os.path.join(dataset_crop, sample["folder"], sample["filename"])
        if not os.path.exists(img_path):
            cls_dir = os.path.join(dataset_crop, sample["folder"])
            if os.path.exists(cls_dir) and len(os.listdir(cls_dir)) > 0:
                img_path = os.path.join(cls_dir, sorted(os.listdir(cls_dir))[0])
            else:
                continue

        img_pil = Image.open(img_path).convert("RGB").resize((224, 224))
        img_np = np.array(img_pil)

        ax_orig = axes[r_idx, 0]
        ax_orig.imshow(img_np)
        if r_idx == 0:
            ax_orig.set_title("Input Asli", fontsize=11, fontweight='bold')
        ax_orig.set_ylabel(f"Kelas: {sample['class_name']}", fontsize=10, fontweight='bold')
        ax_orig.set_xticks([]); ax_orig.set_yticks([])

        input_clean = val_tf(img_pil).unsqueeze(0).to(DEVICE)
        clean_cam, _ = grad_cam(input_clean, class_idx=sample["target_cls"])

        for c_idx, sig in enumerate(sigmas, start=1):
            noisy_np = add_gaussian_noise(img_np, sig)
            noisy_pil = Image.fromarray(noisy_np)
            input_noisy = val_tf(noisy_pil).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                out = model(input_noisy)
                prob = torch.softmax(out, dim=1)[0, sample["target_cls"]].item()

            noisy_cam, _ = grad_cam(input_noisy, class_idx=sample["target_cls"])
            corr = np.corrcoef(clean_cam.flatten(), noisy_cam.flatten())[0, 1]

            results.append({
                'class': sample['class_name'],
                'noise_sigma': sig,
                'confidence': prob * 100,
                'cam_correlation': max(0.0, corr)
            })

            heatmap = cv2.applyColorMap(np.uint8(255 * noisy_cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            overlay = cv2.addWeighted(noisy_np, 0.55, heatmap, 0.45, 0)

            ax = axes[r_idx, c_idx]
            ax.imshow(overlay)
            if r_idx == 0:
                ax.set_title(f"Noise σ={sig}", fontsize=11, fontweight='bold')
            ax.set_xlabel(f"Conf: {prob*100:.1f}%\nCorr: {corr:.2f}", fontsize=8.5)
            ax.set_xticks([]); ax.set_yticks([])

    grad_cam.remove_hooks()

    plt.tight_layout()
    plot_path = os.path.join(OUTPUT_DIR, "evaluasi_gradcam_noise.png")
    plt.savefig(plot_path, bbox_inches='tight', facecolor='#ffffff')
    plt.close()

    df_results = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "hasil_uji_noise.csv")
    df_results.to_csv(csv_path, index=False)

    print("=" * 75)
    print("🏆 HASIL UJI KETAHANAN GRAD-CAM TERHADAP NOISE:")
    print(df_results.to_string(index=False))
    print(f"\n💾 Visualisasi disimpan di : {plot_path}")
    print(f"📄 CSV Detail disimpan di   : {csv_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
