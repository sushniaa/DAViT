import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from collections import Counter
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split

# Transforms
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transforms = A.Compose([
    A.LongestMaxSize(1024, p=1),
    A.PadIfNeeded(min_height=1024, min_width=1024, p=1, border_mode=cv2.BORDER_REFLECT_101, value=0),
    A.RandomResizedCrop(height=224, width=224, scale=(0.8, 1.0), ratio=(0.9, 1.1), interpolation=cv2.INTER_CUBIC, p=1),
    A.Rotate(limit=5, p=0.5),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.Affine(scale=(0.98, 1.02), rotate=(-5, 5), shear=(-2, 2), p=0.5),
    A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.05, p=0.5),
    A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
    A.HueSaturationValue(hue_shift_limit=1, sat_shift_limit=10, val_shift_limit=15, p=0.5),
    A.ShiftScaleRotate(shift_limit=0.005, scale_limit=0.005, rotate_limit=10, border_mode=cv2.BORDER_REFLECT, p=0.5),
    A.OneOf([
        A.Blur(blur_limit=3, p=0.3),
        A.GaussNoise(var_limit=(5.0, 10.0), p=0.3),
    ], p=0.5),
    A.Normalize(mean=mean, std=std),
    ToTensorV2(),
])

val_transforms = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=mean, std=std),
    ToTensorV2()
])

test_transforms = val_transforms

class SkinLesionDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        if isinstance(csv_file, pd.DataFrame):
            self.data = csv_file
        else:
            self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.domain_map = {
            'torso': 0, 'lower extremity': 1, 'upper extremity': 2, 'anterior torso': 3,
            'head/neck': 4, 'posterior torso': 5, 'palms/soles': 6, 'oral/genital': 7,
            'lateral torso': 8, 'unknown': 9
        }
        self.diagnosis_to_target = {
            'NV': 5, 'MEL': 4, 'BKL': 2, 'DF': 3, 'SCC': 6, 'BCC': 1, 'VASC': 8, 'AK': 0, 'UNK': 7
        }

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['path']
        if not os.path.exists(img_path):
            raise ValueError(f"Error: Image at {img_path} not found.")
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Error: Unable to read image at {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.data.iloc[idx]['target'])
        anatom_site = str(self.data.iloc[idx]['anatom_site_general']).lower()
        domain = self.domain_map.get(anatom_site, 9)
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]
        return image, label, domain

def collate_fn(batch):
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    domains = torch.tensor([item[2] for item in batch], dtype=torch.long)
    return images, labels, domains

# Data Loaders
def create_data_loaders(csv_path):
    df = pd.read_csv(csv_path)
    df["anatom_site_general"] = df["anatom_site_general"].fillna("unknown")
    domain_labels = df["anatom_site_general"].values
    train_val_df, test_df = train_test_split(df, test_size=0.1, stratify=domain_labels, random_state=42)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, stratify=train_val_df["anatom_site_general"], random_state=42)
    print(f"Train Size: {len(train_df)}, Val Size: {len(val_df)}, Test Size: {len(test_df)}")

    train_dataset = SkinLesionDataset(train_df, transform=train_transforms)
    val_dataset = SkinLesionDataset(val_df, transform=val_transforms)
    test_dataset = SkinLesionDataset(test_df, transform=test_transforms)

    domain_counts = Counter(train_df["anatom_site_general"])
    total_samples = sum(domain_counts.values())
    weights = {domain: total_samples / count for domain, count in domain_counts.items()}
    train_sample_weights = [weights[domain] for domain in train_df["anatom_site_general"]]
    train_sampler = WeightedRandomSampler(train_sample_weights, num_samples=len(train_sample_weights), replacement=False)

    train_dataloader = DataLoader(train_dataset, batch_size=16, sampler=train_sampler, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=2, pin_memory=True)
    print(len(train_dataloader), len(val_dataloader), len(test_dataloader))
    return train_dataloader, val_dataloader, test_dataloader
