import os
import random
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split


# Standard 5-Class Diagnostic Taxonomy
CLASSES = ("BCC", "MEL", "NV", "AKIEC", "BKL")
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
CLASS_NAMES = {
    "BCC": "Basal Cell Carcinoma",
    "MEL": "Melanoma",
    "NV": "Melanocytic Nevus",
    "AKIEC": "Actinic Keratoses and Intraepithelial Carcinoma (ACK + SCC)",
    "BKL": "Benign Keratosis-like Lesions (Seborrheic Keratosis)",
}

# Mapping from PAD-UFES-20 smartphone categories to shared 5 benchmark classes
PAD_TO_SHARED_CLASS = {
    "BCC": "BCC",
    "MEL": "MEL",
    "NEV": "NV",
    "ACK": "AKIEC",  # Actinic Keratosis (pre-malignant dysplasia)
    "SCC": "AKIEC",  # Squamous Cell Carcinoma (invasive carcinoma)
    "SEK": "BKL",    # Seborrheic Keratosis
}

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class SkinDomainDataset(Dataset):
    """
    Skin lesion dataset loading RGB image files and integer class labels.
    """
    def __init__(self, items: List[Tuple[str, int]], transform=None):
        self.items = items
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.items[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(img_size: int = 288):
    """Constructs train and validation augmentation pipelines."""
    train_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transform, val_transform


def build_domain_splits(
    data_dir: str,
    target_n: Optional[int] = None,
    seed: int = 42
) -> Dict[str, Dict[str, List[Tuple[str, int]]]]:
    """
    Builds stratified splits for Source (ISIC 2018 dermoscopy) and Target (PAD-UFES-20 smartphone).
    Target data is partitioned into:
      - 70% Train (1,608 images)
      - 15% Validation (345 images)
      - 15% Held-Out Test (345 images)
    If target_n is specified (e.g. 50, 100, 200), the target training set is subsampled
    using stratified sampling while preserving the identical validation and held-out test sets.
    """
    # 1. Source (ISIC 2018)
    isic_train_csv = os.path.join(data_dir, "ISIC2018_Task3_Training_GroundTruth", "ISIC2018_Task3_Training_GroundTruth.csv")
    isic_train_dir = os.path.join(data_dir, "ISIC2018_Task3_Training_Input")
    isic_test_csv = os.path.join(data_dir, "ISIC2018_Task3_Test_GroundTruth", "ISIC2018_Task3_Test_GroundTruth.csv")
    isic_test_dir = os.path.join(data_dir, "ISIC2018_Task3_Test_Input")

    df_src_train = pd.read_csv(isic_train_csv)
    source_train_full = []
    for _, row in df_src_train.iterrows():
        fname = f"{row['image']}.jpg"
        fpath = os.path.join(isic_train_dir, fname)
        if os.path.isfile(fpath):
            for c in CLASSES:
                if c in row and row[c] == 1.0:
                    source_train_full.append((fpath, CLASS_TO_IDX[c]))
                    break

    # Stratified 85/15 source train/val split
    src_y = [x[1] for x in source_train_full]
    src_train_items, src_val_items = train_test_split(
        source_train_full, test_size=0.15, random_state=seed, stratify=src_y
    )

    source_test_items = []
    if os.path.isfile(isic_test_csv) and os.path.isdir(isic_test_dir):
        df_src_test = pd.read_csv(isic_test_csv)
        for _, row in df_src_test.iterrows():
            fname = f"{row['image']}.jpg"
            fpath = os.path.join(isic_test_dir, fname)
            if os.path.isfile(fpath):
                for c in CLASSES:
                    if c in row and row[c] == 1.0:
                        source_test_items.append((fpath, CLASS_TO_IDX[c]))
                        break

    # 2. Target (PAD-UFES-20)
    pad_csv = os.path.join(data_dir, "metadata.csv")
    pad_dirs = [
        os.path.join(data_dir, "images", "imgs_part_1"),
        os.path.join(data_dir, "images", "imgs_part_2"),
        os.path.join(data_dir, "images", "imgs_part_3"),
    ]
    pad_map = {}
    for d in pad_dirs:
        if os.path.isdir(d):
            for f in os.listdir(d):
                pad_map[f] = os.path.join(d, f)

    df_pad = pd.read_csv(pad_csv)
    target_items_raw = []
    for _, row in df_pad.iterrows():
        img_id = str(row["img_id"]).strip()
        diag = str(row["diagnostic"]).strip()
        if diag in PAD_TO_SHARED_CLASS and img_id in pad_map:
            shared_c = PAD_TO_SHARED_CLASS[diag]
            target_items_raw.append((pad_map[img_id], CLASS_TO_IDX[shared_c]))

    # Fixed 70/15/15 stratified split
    tgt_y = [x[1] for x in target_items_raw]
    tgt_train_full, tgt_remain = train_test_split(
        target_items_raw, test_size=0.30, random_state=42, stratify=tgt_y
    )
    remain_y = [x[1] for x in tgt_remain]
    tgt_val_items, tgt_test_items = train_test_split(
        tgt_remain, test_size=0.50, random_state=42, stratify=remain_y
    )

    # Subsample target train if target_n is specified
    if target_n is not None and target_n < len(tgt_train_full):
        full_y = [x[1] for x in tgt_train_full]
        tgt_train_items, _ = train_test_split(
            tgt_train_full, train_size=target_n, random_state=seed, stratify=full_y
        )
    else:
        tgt_train_items = tgt_train_full

    return {
        "train": {"source": src_train_items, "target": tgt_train_items},
        "val": {"source": src_val_items, "target": tgt_val_items},
        "test": {"source": source_test_items, "target": tgt_test_items},
    }
