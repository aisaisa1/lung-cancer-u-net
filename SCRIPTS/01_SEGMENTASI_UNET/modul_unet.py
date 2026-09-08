import os
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models
import cv2
import numpy as np
from scipy import ndimage

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PACKAGE_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '../..'))
DEVICE = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')

CLASSES = ['Normal cases', 'Malignant cases']
CLASS_NAMES = ['NORMAL', 'CANCER']
IMG_SIZE = 224

# ===================================================
# --- VARIAN 1: RESNET-18 ATTENTION U-NET (PRIMARY) ---
# ===================================================
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=True)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionDecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(AttentionDecoderBlock, self).__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.ag = AttentionGate(F_g=in_channels // 2, F_l=skip_channels, F_int=out_channels // 2)
        self.conv = nn.Sequential(
            nn.Conv2d((in_channels // 2) + skip_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip=None):
        x = self.upsample(x)
        if skip is not None:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            skip = self.ag(g=x, x=skip)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class ResNet18_UNet(nn.Module):
    """
    Varian 1: ResNet18 Attention U-Net (High-Precision Backbone U-Net)
    Digunakan untuk segmentasi utama & cropping ROI citra untuk model klasifikasi.
    """
    def __init__(self, in_ch=3, out_ch=1, pretrained=True):
        super(ResNet18_UNet, self).__init__()
        resnet = models.resnet18(weights='DEFAULT' if pretrained else None)

        if in_ch != 3:
            self.init_conv = nn.Sequential(
                nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False),
                resnet.bn1,
                resnet.relu
            )
        else:
            self.init_conv = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)

        self.maxpool = resnet.maxpool

        # Encoder
        self.layer1 = resnet.layer1 # 64
        self.layer2 = resnet.layer2 # 128
        self.layer3 = resnet.layer3 # 256
        self.layer4 = resnet.layer4 # 512

        # Decoder
        self.dec4 = AttentionDecoderBlock(in_channels=512, skip_channels=256, out_channels=256)
        self.dec3 = AttentionDecoderBlock(in_channels=256, skip_channels=128, out_channels=128)
        self.dec2 = AttentionDecoderBlock(in_channels=128, skip_channels=64, out_channels=64)
        self.dec1 = AttentionDecoderBlock(in_channels=64, skip_channels=64, out_channels=64)

        # Output head
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_ch, kernel_size=1)
        )

    def forward(self, x):
        x0 = self.init_conv(x)
        x_mp = self.maxpool(x0)

        x1 = self.layer1(x_mp)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        d4 = self.dec4(x4, x3)
        d3 = self.dec3(d4, x2)
        d2 = self.dec2(d3, x1)
        d1 = self.dec1(d2, x0)

        out = self.final_up(d1)
        return out


# ===================================================
# --- VARIAN 2: U-NET BASIC (STANDAR VANILLA U-NET) ---
# ===================================================
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet_Basic(nn.Module):
    """
    Varian 2: Vanilla U-Net Standar
    Classic 4-stage Double Conv U-Net
    """

    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()

        # =====================================================
        # ENCODER
        # =====================================================

        self.inc = DoubleConv(in_channels, 64)

        self.down1 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(64, 128)
        )

        self.down2 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(128, 256)
        )

        self.down3 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(256, 512)
        )

        # Bottleneck
        self.down4 = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(512, 1024)
        )

        # =====================================================
        # DECODER
        # =====================================================

        # 1024 -> 512
        self.up1 = nn.ConvTranspose2d(
            1024, 512, kernel_size=2, stride=2
        )

        self.conv_up1 = DoubleConv(
            1024, 512
        )

        # 512 -> 256
        self.up2 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )

        self.conv_up2 = DoubleConv(
            512, 256
        )

        # 256 -> 128
        self.up3 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )

        self.conv_up3 = DoubleConv(
            256, 128
        )

        # 128 -> 64
        self.up4 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )

        self.conv_up4 = DoubleConv(
            128, 64
        )

        # =====================================================
        # OUTPUT
        # =====================================================

        self.outc = nn.Conv2d(
            64,
            out_channels,
            kernel_size=1
        )

    def forward(self, x):

        # =====================================================
        # ENCODER
        # =====================================================

        x1 = self.inc(x)       # 64
        x2 = self.down1(x1)    # 128
        x3 = self.down2(x2)    # 256
        x4 = self.down3(x3)    # 512
        x5 = self.down4(x4)    # 1024

        # =====================================================
        # DECODER
        # =====================================================

        x = self.up1(x5)       # 512
        x = torch.cat([x, x4], dim=1)   # 512 + 512 = 1024
        x = self.conv_up1(x)   # 512

        x = self.up2(x)        # 256
        x = torch.cat([x, x3], dim=1)   # 256 + 256 = 512
        x = self.conv_up2(x)   # 256

        x = self.up3(x)        # 128
        x = torch.cat([x, x2], dim=1)   # 128 + 128 = 256
        x = self.conv_up3(x)   # 128

        x = self.up4(x)        # 64
        x = torch.cat([x, x1], dim=1)   # 64 + 64 = 128
        x = self.conv_up4(x)   # 64

        # =====================================================
        # OUTPUT
        # =====================================================

        return self.outc(x)

# ===================================================
# --- VARIAN 3: U-NET LIGHT (RINGAN & CEPAT) ---
# ===================================================
class UNet_Light(nn.Module):
    """
    Varian 3: U-Net Light (Arsitektur Efisien dengan Fitur Ringan 16 -> 32 -> 64 -> 128)
    """
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(32, 64)
        self.pool3 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(64, 128)

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 64)

        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = DoubleConv(64, 32)

        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = DoubleConv(32, 16)

        self.outc = nn.Conv2d(16, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        b = self.bottleneck(self.pool3(e3))

        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        return self.outc(d1)


UNet = ResNet18_UNet

# ===================================================
# --- HYBRID IOUSUPER LOSS & UTILITAS ---
# ===================================================
def lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / (union + 1e-6)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

def lovasz_hinge_flat(logits, labels):
    if len(labels) == 0:
        return logits.sum() * 0.0
    signs = 2.0 * labels.float() - 1.0
    errors = 1.0 - logits * signs
    errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
    perm = perm.data
    gt_sorted = labels[perm]
    grad = lovasz_grad(gt_sorted)
    loss = torch.dot(F.relu(errors_sorted), grad)
    return loss

def lovasz_hinge_loss(logits, labels):
    return lovasz_hinge_flat(logits.view(-1), labels.view(-1))

class BoundaryContourLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kernel = torch.tensor([[0., 1., 0.],
                               [1., -4., 1.],
                               [0., 1., 0.]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.register_buffer('laplacian', kernel)

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        pred_edge = torch.abs(F.conv2d(probs, self.laplacian, padding=1))
        target_edge = torch.abs(F.conv2d(targets, self.laplacian, padding=1))
        return F.mse_loss(pred_edge, target_edge)

class HybridIoUSuperLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.boundary = BoundaryContourLoss()

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        lovasz_loss = lovasz_hinge_loss(logits, targets)
        boundary_loss = self.boundary(logits, targets)
        return 0.50 * lovasz_loss + 0.30 * bce_loss + 0.20 * boundary_loss

def refine_lung_mask(bin_mask):
    mask_bool = bin_mask > 0
    mask_filled = ndimage.binary_fill_holes(mask_bool)
    mask_uint = (mask_filled * 255).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_closed = cv2.morphologyEx(mask_uint, cv2.MORPH_CLOSE, kernel)
    return (mask_closed > 127).astype(np.uint8)

class LungSegDataset(Dataset):
    def __init__(self, pairs, size=IMG_SIZE, augment=False):
        self.pairs = pairs
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def generate_pseudo_mask(self, img):
        """
        Generate pseudo-mask paru secara otomatis.
        Tidak menggunakan Mask_Manual_GT.
        """

        # RGB -> grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Sedikit blur untuk mengurangi noise
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu threshold
        _, mask = cv2.threshold(
            blur,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Pastikan foreground adalah area yang lebih gelap
        # karena paru biasanya lebih gelap daripada jaringan sekitarnya
        if np.mean(gray[mask > 0]) > np.mean(gray[mask == 0]):
            mask = 255 - mask

        # Morphological opening
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        # Morphological closing
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Fill holes
        mask = refine_lung_mask(mask)

        return mask.astype(np.float32)

    def __getitem__(self, idx):

        # Sekarang pairs hanya berisi image path
        img_path = self.pairs[idx]

        img = cv2.imread(img_path)

        if img is None:
            raise ValueError(
                f"Gagal membaca gambar: {img_path}"
            )

        img = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2RGB
        )

        # Resize image
        img = cv2.resize(
            img,
            (self.size, self.size),
            interpolation=cv2.INTER_AREA
        )

        # =====================================================
        # GENERATE PSEUDO MASK OTOMATIS
        # =====================================================

        mask = self.generate_pseudo_mask(img)

        # =====================================================
        # AUGMENTATION
        # =====================================================

        if self.augment:

            if np.random.rand() > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()

        # =====================================================
        # NORMALISASI IMAGE
        # =====================================================

        img = img.astype(np.float32) / 255.0

        mean = np.array(
            [0.485, 0.456, 0.406],
            dtype=np.float32
        )

        std = np.array(
            [0.229, 0.224, 0.225],
            dtype=np.float32
        )

        img = (img - mean) / std

        img = img.transpose(2, 0, 1)

        # =====================================================
        # RETURN IMAGE + AUTOMATIC PSEUDO MASK
        # =====================================================

        return (
            torch.from_numpy(img),
            torch.from_numpy(mask).unsqueeze(0)
        )

def build_pairs(
    img_subfolder='Citra_Gamma_CLAHE',
    package_root=PACKAGE_ROOT
):
    input_root = os.path.join(
        package_root,
        'DATASET',
        img_subfolder
    )

    print("\n" + "=" * 70)
    print("📂 DATASET CHECK")
    print("=" * 70)
    print(f"PACKAGE ROOT : {package_root}")
    print(f"IMAGE ROOT   : {input_root}")
    print(f"EXISTS       : {os.path.exists(input_root)}")

    if not os.path.exists(input_root):
        print("❌ Dataset tidak ditemukan:")
        print(input_root)
        return {}

    pairs_by_class = {}

    for cls in CLASSES:
        in_dir = os.path.join(input_root, cls)

        print(f"\n📁 CLASS: {cls}")
        print(f"PATH   : {in_dir}")
        print(f"EXISTS : {os.path.exists(in_dir)}")

        if not os.path.isdir(in_dir):
            continue

        files = sorted([
            f for f in os.listdir(in_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        pairs_by_class[cls] = [
            os.path.join(in_dir, f)
            for f in files
        ]

        print(f"📸 JUMLAH CITRA: {len(files)}")

    print("=" * 70)

    return pairs_by_class

def dice_coef(gt, pred, smooth=1e-6):
    gt = gt > 0
    pred = pred > 0
    inter = np.logical_and(gt, pred).sum()
    total = gt.sum() + pred.sum()
    if total == 0:
        return 1.0
    return (2.0 * inter + smooth) / (total + smooth)

def iou_score(gt, pred, smooth=1e-6):
    gt = gt > 0
    pred = pred > 0
    union = np.logical_or(gt, pred).sum()
    inter = np.logical_and(gt, pred).sum()
    if union == 0:
        return 1.0
    return (inter + smooth) / (union + smooth)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
