# dataset_creation_isic.py

import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split

# =========================
# PATHS
# =========================
ISIC_2019_DIR = '/content/2019/train'
ISIC_2020_DIR = '/content/2020/train'
OUTPUT_DIR = 'results'

# =========================
# UTILS
# =========================
def to_path(image_name, path):
    return os.path.join(path, f"{image_name}.jpg")

def mapping_2019(df):
    df['diagnosis'] = None
    df['diagnosis'].mask(df['MEL'] == 1, 'MEL', inplace=True)
    df['diagnosis'].mask(df['NV'] == 1, 'NV', inplace=True)
    df['diagnosis'].mask(df['BCC'] == 1, 'BCC', inplace=True)
    df['diagnosis'].mask(df['AK'] == 1, 'AK', inplace=True)
    df['diagnosis'].mask(df['BKL'] == 1, 'BKL', inplace=True)
    df['diagnosis'].mask(df['DF'] == 1, 'DF', inplace=True)
    df['diagnosis'].mask(df['VASC'] == 1, 'VASC', inplace=True)
    df['diagnosis'].mask(df['SCC'] == 1, 'SCC', inplace=True)
    df['diagnosis'].mask(df['UNK'] == 1, 'UNK', inplace=True)
    return df

def mapping_2020(df):
    replace_map = {
        'seborrheic keratosis': 'BKL',
        'lichenoid keratosis': 'BKL',
        'solar lentigo': 'BKL',
        'lentigo NOS': 'BKL',
        'cafe-au-lait macule': 'UNK',
        'atypical melanocytic proliferation': 'UNK',
        'unknown': 'UNK',
        'nevus': 'NV',
        'melanoma': 'MEL'
    }
    df['diagnosis'] = df['diagnosis'].replace(replace_map)
    return df

def create_target(df):
    diagnosis_idx = {d: idx for idx, d in enumerate(sorted(df.diagnosis.unique()))}
    df['target'] = df['diagnosis'].map(diagnosis_idx)
    return df

def create_isic_df(dir_path, csv_file, date):
    df = pd.read_csv(os.path.join(dir_path, csv_file))
    if date == '2020':
        df = df.rename(columns={"image_name": "image"})
    df['path'] = df['image'].apply(lambda x: to_path(x, dir_path))

    if date == '2019':
        df = mapping_2019(df)
    elif date == '2020':
        df = mapping_2020(df)

    keep_cols = ['image', 'path', 'diagnosis', 'age_approx', 'anatom_site_general', 'sex']
    df = df[[c for c in df.columns if c in keep_cols]]
    df = create_target(df)
    return df

# =========================
# PHASH DUPLICATE REMOVAL
# =========================
def compute_phash(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (32, 32))
    dct = cv2.dct(np.float32(img))
    dct_low_freq = dct[:8, :8]
    med = np.median(dct_low_freq)
    hash_bin = (dct_low_freq > med).astype(int)
    return ''.join(map(str, hash_bin.flatten()))

def remove_duplicates(df):
    hash_dict, duplicates = {}, set()
    for _, row in df.iterrows():
        img_path = row['path']
        if os.path.exists(img_path):
            h = compute_phash(img_path)
            if h:
                if h in hash_dict:
                    duplicates.add(row['image'] + '.jpg')
                else:
                    hash_dict[h] = row['image'] + '.jpg'
    df['temporal_image'] = df['image'] + '.jpg'
    df_clean = df[~df['temporal_image'].isin(duplicates)].drop(columns=['temporal_image'])
    return df_clean

# =========================
# MAIN
# =========================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load datasets
    df_2019 = create_isic_df(ISIC_2019_DIR, 'train.csv', '2019')
    df_2020 = create_isic_df(ISIC_2020_DIR, 'train.csv', '2020')

    # Combine
    df_combined = pd.concat([df_2019, df_2020], ignore_index=True, sort=False)
    print("Before duplicate removal:", df_combined.shape)

    # Remove duplicates
    df_clean = remove_duplicates(df_combined)
    print("After duplicate removal:", df_clean.shape)
    print("Diagnosis counts:\n", df_clean['diagnosis'].value_counts())

    # Save whole dataset
    df_clean.to_csv(os.path.join(OUTPUT_DIR, 'whole_data_no_duplicates.csv'), index=False)

    # Split train/val
    df_train, df_val = train_test_split(
        df_clean, test_size=0.136, stratify=df_clean['target'], random_state=42
    )
    df_train.to_csv(os.path.join(OUTPUT_DIR, 'train_whole_no_duplicates.csv'), index=False)
    df_val.to_csv(os.path.join(OUTPUT_DIR, 'val_whole_no_duplicates.csv'), index=False)

    print(f"Train size: {len(df_train)}, Val size: {len(df_val)}")

if __name__ == "__main__":
    main()
