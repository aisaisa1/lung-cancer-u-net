import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
KLASIFIKASI_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '../02_KLASIFIKASI_MODEL'))
sys.path.insert(0, KLASIFIKASI_DIR)
from modul_klasifikasi import ResNet18_CBAM, DEVICE, PACKAGE_ROOT

OUTPUT_DIR = os.path.join(PACKAGE_ROOT, 'OUTPUT_HASIL')
os.makedirs(OUTPUT_DIR, exist_ok=True)

class QuantusGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []
        self.hook_handles.append(self.target_layer.register_forward_hook(self._save_activation))
        self.hook_handles.append(self.target_layer.register_full_backward_hook(self._save_gradient))

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        for h in self.hook_handles: h.remove()

    def __call__(self, input_tensor, target_class=None, apply_mask=True):
        self.model.eval()
        self.model.zero_grad()
        output = self.model(input_tensor)
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        prob = torch.softmax(output, dim=1)[0, target_class].item()
        output[0, target_class].backward(retain_graph=True)
        
        grads = self.gradients[0].cpu().numpy()
        acts = self.activations[0].cpu().numpy()
        weights = np.mean(grads, axis=(1, 2))
        
        cam = np.zeros(acts.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * acts[i, :, :]
            
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224), interpolation=cv2.INTER_CUBIC)
        
        if apply_mask:
            img_np = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
            img_gray = np.mean(img_np, axis=2)
            img_gray = (img_gray - img_gray.min()) / (img_gray.max() - img_gray.min() + 1e-8)
            mask = (img_gray > 0.08).astype(np.float32)
            cam = cam * mask
            
        cam = cv2.GaussianBlur(cam, (21, 21), 0)
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        return cam, target_class, prob

def create_overlay(orig_img_rgb, cam_heatmap, colormap=cv2.COLORMAP_JET, alpha=0.45):
    heatmap_uint8 = np.uint8(255 * cam_heatmap)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    orig_uint8 = np.array(orig_img_rgb)
    return cv2.addWeighted(orig_uint8, 1 - alpha, heatmap_color, alpha, 0)

def main():
    print("=" * 75)
    print("🔬 UJI LAYER GRAD-CAM (LAYER 1, LAYER 2, LAYER 3, LAYER 4 + CBAM)")
    print("=" * 75)

    ckpt_path = os.path.join(OUTPUT_DIR, "best_classifier.pt")
    model = ResNet18_CBAM(num_classes=2, pretrained=False).to(DEVICE)
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        print(f"✅ Loaded Model: {ckpt_path}")
    else:
        print("⚠️ Model checkpoint classifier tidak ditemukan, menggunakan weight default.")

    model.eval()

    val_transform = transforms.Compose([
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

    cams_layer = {
        "Layer 1": QuantusGradCAM(model, model.cbam1),
        "Layer 2": QuantusGradCAM(model, model.cbam2),
        "Layer 3": QuantusGradCAM(model, model.cbam3),
        "Layer 4": QuantusGradCAM(model, model.cbam4)
    }

    col_headers = ["Input Crop (ROI)", "Layer 1 + CBAM", "Layer 2 + CBAM", "Layer 3 + CBAM", "Layer 4 + CBAM"]
    total_rows = len(samples)
    num_cols = len(col_headers)

    fig, axes = plt.subplots(total_rows, num_cols, figsize=(15, 3.2 * total_rows), dpi=200)

    for r_idx, sample in enumerate(samples):
        img_path = os.path.join(dataset_crop, sample["folder"], sample["filename"])
        if not os.path.exists(img_path):
            cls_dir = os.path.join(dataset_crop, sample["folder"])
            if os.path.exists(cls_dir) and len(os.listdir(cls_dir)) > 0:
                img_path = os.path.join(cls_dir, sorted(os.listdir(cls_dir))[0])
            else:
                continue

        img_pil = Image.open(img_path).convert("RGB").resize((224, 224))
        input_tensor = val_transform(img_pil).unsqueeze(0).to(DEVICE)

        ax_input = axes[r_idx, 0]
        ax_input.imshow(img_pil)
        if r_idx == 0:
            ax_input.set_title(col_headers[0], fontsize=11, fontweight='bold')
        ax_input.set_ylabel(f"Kelas: {sample['class_name']}", fontsize=10, fontweight='bold')
        ax_input.set_xticks([]); ax_input.set_yticks([])

        for col_idx, (layer_name, cam_extractor) in enumerate(cams_layer.items(), start=1):
            ax = axes[r_idx, col_idx]
            heatmap, _, prob = cam_extractor(input_tensor, target_class=sample["target_cls"])
            overlay = create_overlay(img_pil, heatmap, alpha=0.45)
            ax.imshow(overlay)
            if r_idx == 0:
                ax.set_title(col_headers[col_idx], fontsize=11, fontweight='bold')
            ax.set_xlabel(f"Conf: {prob*100:.1f}%", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

    for cam in cams_layer.values():
        cam.remove_hooks()

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "perbandingan_layer_gradcam.png")
    plt.savefig(save_path, bbox_inches='tight', facecolor='#ffffff')
    plt.close()

    print(f"🎉 UJI LAYER GRAD-CAM SELESAI! Gambar disimpan ke:\n   📁 {save_path}")
    print("=" * 75)

if __name__ == "__main__":
    main()
