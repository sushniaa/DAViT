import os
import shutil
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, recall_score,
    f1_score, roc_auc_score, roc_curve, confusion_matrix,
    classification_report, auc
)
from sklearn.preprocessing import label_binarize

import albumentations as A
from albumentations.pytorch import ToTensorV2

from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import ViTModel

def check_device():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print("CUDA Available:", torch.cuda.is_available())
    print("CUDA Version:", torch.version.cuda)
    print("cuDNN Version:", torch.backends.cudnn.version())
    print("Device Count:", torch.cuda.device_count())
    print("Current Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
    os.system('nvidia-smi')

def denormalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    if isinstance(image, torch.Tensor):
        image = image.numpy().transpose(1, 2, 0)
    image = image * std + mean
    image = np.clip(image, 0, 1)
    return image

def visualize_augmentations(dataset, num_images=10):
    fig, axes = plt.subplots(num_images, 2, figsize=(15, num_images * 2.5))
    for i in range(num_images):
        idx = np.random.randint(0, len(dataset) - 1)
        img_path = dataset.data.iloc[idx]['path']
        original_image = cv2.imread(img_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        augmented_image, label, domain = dataset[idx]
        augmented_image = denormalize(augmented_image)
        axes[i, 0].imshow(original_image)
        axes[i, 0].axis("off")
        axes[i, 0].set_title(f"Original\nLabel: {label}, Domain: {domain}")
        axes[i, 1].imshow(augmented_image)
        axes[i, 1].axis("off")
        axes[i, 1].set_title(f"Augmented\nLabel: {label}, Domain: {domain}")
    plt.subplots_adjust(hspace=0.5)
    plt.tight_layout()
    plt.show()

def plot_training_dynamics(train_losses, val_losses, train_accuracies, val_accuracies):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.show()
