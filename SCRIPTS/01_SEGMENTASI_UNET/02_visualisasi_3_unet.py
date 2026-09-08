import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from modul_unet import (
    ResNet18_UNet,
    UNet_Basic,
    UNet_Light,
    LungSegDataset,
    build_pairs,
    DEVICE,
    IMG_SIZE,
    PACKAGE_ROOT,
    CLASSES
)


# ============================================================
# CONFIG
# ============================================================

OUTPUT_DIR = os.path.join(
    PACKAGE_ROOT,
    "OUTPUT_HASIL",
    "VISUALISASI_3_UNET"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_DIR = os.path.join(
    PACKAGE_ROOT,
    "OUTPUT_HASIL"
)

MODEL_PATHS = {
    "ResNet18 Attention U-Net": os.path.join(
        MODEL_DIR,
        "best_unet_resnet18_attention_u_net.pt"
    ),

    "U-Net Basic": os.path.join(
        MODEL_DIR,
        "best_unet_u_net_basic.pt"
    ),

    "U-Net Light": os.path.join(
        MODEL_DIR,
        "best_unet_u_net_light.pt"
    )
}


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_checkpoint(model, path):
    print("\nLoading:", os.path.basename(path))

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint tidak ditemukan:\n{path}"
        )

    checkpoint = torch.load(
        path,
        map_location=DEVICE
    )

    # Jika checkpoint berupa dictionary
    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]

        else:
            # Kemungkinan langsung state_dict
            state_dict = checkpoint

    else:
        state_dict = checkpoint

    # Hilangkan prefix module. jika ada
    new_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):
            key = key.replace("module.", "", 1)

        new_state_dict[key] = value

    model.load_state_dict(
        new_state_dict,
        strict=True
    )

    model.to(DEVICE)
    model.eval()

    print("✓ Model berhasil dimuat")

    return model


# ============================================================
# CREATE MODELS
# ============================================================

def create_models():

    print("\n" + "=" * 70)
    print("LOAD 3 MODEL U-NET")
    print("=" * 70)

    # pretrained=False supaya tidak download ResNet lagi.
    resnet_model = ResNet18_UNet(
        in_ch=3,
        out_ch=1,
        pretrained=False
    )

    basic_model = UNet_Basic(
        in_channels=3,
        out_channels=1
    )

    light_model = UNet_Light(
        in_channels=3,
        out_channels=1
    )

    models_dict = {}

    models_dict["ResNet18 Attention U-Net"] = load_checkpoint(
        resnet_model,
        MODEL_PATHS["ResNet18 Attention U-Net"]
    )

    models_dict["U-Net Basic"] = load_checkpoint(
        basic_model,
        MODEL_PATHS["U-Net Basic"]
    )

    models_dict["U-Net Light"] = load_checkpoint(
        light_model,
        MODEL_PATHS["U-Net Light"]
    )

    print("\n✓ SEMUA MODEL BERHASIL DIMUAT")

    return models_dict


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(img_rgb):

    img = cv2.resize(
        img_rgb,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    img_float = img.astype(np.float32) / 255.0

    mean = np.array(
        [0.485, 0.456, 0.406],
        dtype=np.float32
    )

    std = np.array(
        [0.229, 0.224, 0.225],
        dtype=np.float32
    )

    img_norm = (img_float - mean) / std

    img_tensor = img_norm.transpose(2, 0, 1)

    img_tensor = torch.from_numpy(
        img_tensor
    ).unsqueeze(0).float()

    return img, img_tensor.to(DEVICE)


# ============================================================
# PREDICTION
# ============================================================

def predict_mask(model, tensor):

    with torch.no_grad():

        output = model(tensor)

        probability = torch.sigmoid(output)

        prediction = (
            probability > 0.5
        ).float()

    mask = prediction.squeeze().cpu().numpy()

    return mask


# ============================================================
# IOU
# ============================================================

def calculate_iou(gt, pred):

    gt = gt > 0
    pred = pred > 0

    intersection = np.logical_and(
        gt,
        pred
    ).sum()

    union = np.logical_or(
        gt,
        pred
    ).sum()

    if union == 0:
        return 1.0

    return intersection / union


# ============================================================
# LOAD IMAGE + PSEUDO MASK
# ============================================================

def load_image_and_pseudo_mask(image_path):

    img_bgr = cv2.imread(image_path)

    if img_bgr is None:
        raise ValueError(
            f"Gagal membaca gambar:\n{image_path}"
        )

    img_rgb = cv2.cvtColor(
        img_bgr,
        cv2.COLOR_BGR2RGB
    )

    # Dataset hanya digunakan untuk
    # generate pseudo-mask yang sama
    temp_dataset = LungSegDataset(
        [image_path],
        size=IMG_SIZE,
        augment=False
    )

    image_tensor, mask_tensor = temp_dataset[0]

    pseudo_mask = mask_tensor.squeeze().numpy()

    resized_img = cv2.resize(
        img_rgb,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return resized_img, image_tensor.unsqueeze(0).to(DEVICE), pseudo_mask


# ============================================================
# VISUALISASI SATU SAMPLE
# ============================================================

def visualize_sample(
    image_path,
    models_dict,
    save_path,
    sample_number,
    class_name
):

    print(
        f"\n[{sample_number}] "
        f"{class_name} - "
        f"{os.path.basename(image_path)}"
    )

    image, tensor, pseudo_mask = load_image_and_pseudo_mask(
        image_path
    )

    predictions = {}

    for model_name, model in models_dict.items():

        pred = predict_mask(
            model,
            tensor
        )

        predictions[model_name] = pred

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    fig, axes = plt.subplots(
        1,
        5,
        figsize=(20, 4)
    )

    # Original
    axes[0].imshow(image)
    axes[0].set_title("Original")
    axes[0].axis("off")

    # Pseudo mask
    axes[1].imshow(
        pseudo_mask,
        cmap="gray"
    )

    axes[1].set_title(
        "Pseudo Mask\n(Otsu)"
    )

    axes[1].axis("off")

    # ResNet18
    pred = predictions[
        "ResNet18 Attention U-Net"
    ]

    iou = calculate_iou(
        pseudo_mask,
        pred
    )

    axes[2].imshow(
        pred,
        cmap="gray"
    )

    axes[2].set_title(
        f"ResNet18 Attention\nIoU={iou:.2%}"
    )

    axes[2].axis("off")

    # Basic
    pred = predictions[
        "U-Net Basic"
    ]

    iou = calculate_iou(
        pseudo_mask,
        pred
    )

    axes[3].imshow(
        pred,
        cmap="gray"
    )

    axes[3].set_title(
        f"U-Net Basic\nIoU={iou:.2%}"
    )

    axes[3].axis("off")

    # Light
    pred = predictions[
        "U-Net Light"
    ]

    iou = calculate_iou(
        pseudo_mask,
        pred
    )

    axes[4].imshow(
        pred,
        cmap="gray"
    )

    axes[4].set_title(
        f"U-Net Light\nIoU={iou:.2%}"
    )

    axes[4].axis("off")

    fig.suptitle(
        f"{class_name} | {os.path.basename(image_path)}",
        fontsize=13
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "✓ Saved:",
        save_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n")
    print("=" * 70)
    print("VISUALISASI 3 VARIAN U-NET")
    print("=" * 70)

    print("\nDEVICE:", DEVICE)

    # --------------------------------------------------------
    # LOAD MODELS
    # --------------------------------------------------------

    models_dict = create_models()

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    pairs_by_class = build_pairs(
        img_subfolder="Citra_Gamma_CLAHE",
        package_root=PACKAGE_ROOT
    )

    # --------------------------------------------------------
    # SAMPLE
    # --------------------------------------------------------

    NUM_SAMPLES_PER_CLASS = 5

    sample_counter = 1

    for class_name in CLASSES:

        print("\n")
        print("=" * 70)
        print("CLASS:", class_name)
        print("=" * 70)

        image_paths = pairs_by_class.get(
            class_name,
            []
        )

        if len(image_paths) == 0:
            print("Tidak ada gambar.")
            continue

        # Ambil 5 sample pertama
        samples = image_paths[
            :NUM_SAMPLES_PER_CLASS
        ]

        for image_path in samples:

            filename = os.path.splitext(
                os.path.basename(image_path)
            )[0]

            output_filename = (
                f"{sample_counter:02d}_"
                f"{class_name.replace(' ', '_')}_"
                f"{filename}.png"
            )

            save_path = os.path.join(
                OUTPUT_DIR,
                output_filename
            )

            visualize_sample(
                image_path,
                models_dict,
                save_path,
                sample_counter,
                class_name
            )

            sample_counter += 1

    # --------------------------------------------------------
    # DONE
    # --------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("🎉 VISUALISASI SELESAI")
    print("=" * 70)

    print("\nHasil disimpan di:")

    print(
        OUTPUT_DIR
    )

    print("\nIsi folder tersebut akan berupa:")
    print("  01_Normal_cases_....png")
    print("  02_Normal_cases_....png")
    print("  ...")
    print("  06_Malignant_cases_....png")
    print("  ...")


if __name__ == "__main__":
    main()