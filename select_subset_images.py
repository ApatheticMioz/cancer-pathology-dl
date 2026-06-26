#!/usr/bin/env python3
"""
Select a subset of 10 varied images from pannuke, tcga, and siim datasets.
Images are paired with their masks and labels, organized in a structured folder.
"""

import os
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

def create_output_directory(base_path):
    """Create the output directory structure."""
    output_root = Path(base_path) / "subset_10_images"
    if output_root.exists():
        shutil.rmtree(output_root)
    
    datasets = ['pannuke', 'tcga', 'siim']
    for dataset in datasets:
        dataset_dir = output_root / dataset
        (dataset_dir / "images").mkdir(parents=True, exist_ok=True)
        (dataset_dir / "masks").mkdir(parents=True, exist_ok=True)
    
    return output_root


def select_pannuke_images(base_path, output_path, num_images=10):
    """Select varied pannuke images using macenko processed images."""
    print("\n" + "="*60)
    print("PANNUKE DATASET - Macenko Processed Images")
    print("="*60)
    
    # Read the index from preprocessed (which has labels)
    index_path = Path(base_path) / "pannuke" / "preprocessed" / "index.csv"
    df = pd.read_csv(index_path)
    
    # Get unique labels
    unique_labels = sorted(df['label_int'].unique())
    print(f"Unique labels in pannuke: {unique_labels}")
    print(f"Total images available: {len(df)}")
    
    # Select images to maximize label diversity
    selected_images = []
    label_count = defaultdict(int)
    
    # First pass: select one image per label
    for label in unique_labels:
        label_images = df[df['label_int'] == label]
        if len(label_images) > 0:
            # Pick one image from this label
            selected_row = label_images.iloc[0]
            selected_images.append({
                'image_name': Path(selected_row['image_path']).name,
                'label': selected_row['label_int'],
                'group_id': selected_row['group_id']
            })
            label_count[label] += 1
    
    # Second pass: fill remaining slots with images from most common labels
    if len(selected_images) < num_images:
        remaining = num_images - len(selected_images)
        remaining_indices = np.random.choice(
            df.index[~df.index.isin([
                df[df['group_id'] == img['group_id']].index[0] 
                for img in selected_images
            ])],
            size=min(remaining, len(df) - len(selected_images)),
            replace=False
        )
        
        for idx in remaining_indices:
            row = df.iloc[idx]
            selected_images.append({
                'image_name': Path(row['image_path']).name,
                'label': row['label_int'],
                'group_id': row['group_id']
            })
            label_count[row['label_int']] += 1
    
    # Select the first num_images
    selected_images = selected_images[:num_images]
    
    print(f"\nSelected {len(selected_images)} images")
    print("Label distribution:")
    for label in sorted(label_count.keys()):
        print(f"  Label {label}: {label_count[label]} images")
    
    # Copy images and masks - use macenko processed images
    macenko_images_dir = Path(base_path) / "pannuke" / "preprocessed_macenko" / "images"
    preprocessed_images_dir = Path(base_path) / "pannuke" / "preprocessed" / "images"
    preprocessed_masks_dir = Path(base_path) / "pannuke" / "preprocessed" / "masks"
    
    output_dataset_dir = output_path / "pannuke"
    
    for i, img_info in enumerate(selected_images):
        image_name = img_info['image_name']
        label = img_info['label']
        
        # Check if macenko version exists, otherwise use regular preprocessed
        macenko_image_path = macenko_images_dir / image_name
        if macenko_image_path.exists():
            src_image = macenko_image_path
        else:
            src_image = preprocessed_images_dir / image_name
        
        src_mask = preprocessed_masks_dir / image_name
        
        # Create output filename with label
        output_name = f"{i:02d}_label_{label}_{image_name}"
        
        if src_image.exists() and src_mask.exists():
            shutil.copy(src_image, output_dataset_dir / "images" / output_name)
            shutil.copy(src_mask, output_dataset_dir / "masks" / output_name)
            print(f"  Copied: {output_name}")
        else:
            print(f"  WARNING: Missing files for {image_name}")
    
    return selected_images


def select_tcga_images(base_path, output_path, num_images=10):
    """Select varied TCGA images."""
    print("\n" + "="*60)
    print("TCGA DATASET")
    print("="*60)
    
    # TCGA data is organized by patient folders
    tcga_base = Path(base_path) / "TCGA"
    patient_dirs = sorted([d for d in tcga_base.iterdir() if d.is_dir() and d.name.startswith('TCGA_')])
    
    print(f"Found {len(patient_dirs)} patient folders")
    
    # Collect all images across patients
    all_images = []
    for patient_dir in patient_dirs:
        # Get all .tif files (excluding _mask files)
        tif_files = [f for f in patient_dir.glob('*.tif') if '_mask' not in f.name]
        for tif_file in tif_files:
            all_images.append({
                'patient': patient_dir.name,
                'image_name': tif_file.name,
                'image_path': tif_file,
                'mask_path': patient_dir / f"{tif_file.stem}_mask.tif"
            })
    
    print(f"Total TCGA images found: {len(all_images)}")
    
    # Select images from different patients for diversity
    selected_by_patient = defaultdict(list)
    selected_images = []
    
    for img_info in all_images:
        patient = img_info['patient']
        if len(selected_by_patient[patient]) < 2:  # Max 2 images per patient
            selected_by_patient[patient].append(img_info)
            if len(selected_images) < num_images:
                selected_images.append(img_info)
    
    # If we need more images, add from any patient
    if len(selected_images) < num_images:
        remaining_needed = num_images - len(selected_images)
        used_indices = set()
        for images in selected_by_patient.values():
            for img in images:
                # Find index in all_images
                for idx, all_img in enumerate(all_images):
                    if all_img['image_name'] == img['image_name']:
                        used_indices.add(idx)
        
        available = [i for i in range(len(all_images)) if i not in used_indices]
        selected_indices = np.random.choice(available, size=min(remaining_needed, len(available)), replace=False)
        for idx in selected_indices:
            selected_images.append(all_images[idx])
    
    selected_images = selected_images[:num_images]
    
    print(f"\nSelected {len(selected_images)} images from {len(selected_by_patient)} patients")
    print("Patient distribution:")
    for patient in sorted(selected_by_patient.keys()):
        count = len(selected_by_patient[patient])
        if count > 0:
            print(f"  {patient}: {count} images")
    
    # Copy images and masks
    output_dataset_dir = output_path / "tcga"
    
    for i, img_info in enumerate(selected_images):
        src_image = img_info['image_path']
        src_mask = img_info['mask_path']
        
        # Create output filename with patient info
        patient_short = img_info['patient'].replace('TCGA_', '')
        output_name = f"{i:02d}_{patient_short}_{img_info['image_name']}"
        
        if src_image.exists() and src_mask.exists():
            shutil.copy(src_image, output_dataset_dir / "images" / output_name)
            shutil.copy(src_mask, output_dataset_dir / "masks" / output_name)
            print(f"  Copied: {output_name}")
        else:
            print(f"  WARNING: Missing files for {img_info['image_name']}")
    
    return selected_images


def select_siim_images(base_path, output_path, num_images=10):
    """Select varied SIIM images (binary classification: 0=no pneumothorax, 1=pneumothorax)."""
    print("\n" + "="*60)
    print("SIIM DATASET - Pneumothorax Classification")
    print("="*60)
    
    # Read the index
    index_path = Path(base_path) / "SIIM_raw" / "preprocessed" / "index.csv"
    df = pd.read_csv(index_path)
    
    # Get unique labels
    unique_labels = sorted(df['label_int'].unique())
    print(f"Unique labels in SIIM: {unique_labels}")
    print(f"  0 = No pneumothorax")
    print(f"  1 = Pneumothorax")
    print(f"Total images available: {len(df)}")
    
    # Select images to balance labels
    selected_images = []
    label_count = defaultdict(int)
    
    # First pass: select images from each label to balance
    for label in unique_labels:
        label_images = df[df['label_int'] == label]
        # Select half (or close to half) from each label
        select_count = max(1, (num_images // len(unique_labels)) + (1 if label == unique_labels[0] else 0))
        selected_indices = np.random.choice(label_images.index, size=min(select_count, len(label_images)), replace=False)
        
        for idx in selected_indices:
            row = df.iloc[idx]
            if len(selected_images) < num_images:
                selected_images.append({
                    'image_id': row['image_id'],
                    'image_path': row['image_path'],
                    'mask_path': row['mask_path'],
                    'label': row['label_int']
                })
                label_count[label] += 1
    
    print(f"\nSelected {len(selected_images)} images")
    print("Label distribution:")
    for label in sorted(label_count.keys()):
        print(f"  Label {label}: {label_count[label]} images")
    
    # Copy images and masks
    output_dataset_dir = output_path / "siim"
    images_dir = Path(base_path) / "SIIM_raw" / "preprocessed" / "images"
    masks_dir = Path(base_path) / "SIIM_raw" / "preprocessed" / "masks"
    
    for i, img_info in enumerate(selected_images):
        image_name = Path(img_info['image_path']).name
        mask_name = Path(img_info['mask_path']).name
        label = img_info['label']
        
        src_image = images_dir / image_name
        src_mask = masks_dir / mask_name
        
        # Create output filename with label
        label_text = "normal" if label == 0 else "pneumothorax"
        output_name = f"{i:02d}_label_{label}_{label_text}_{image_name}"
        
        if src_image.exists() and src_mask.exists():
            shutil.copy(src_image, output_dataset_dir / "images" / output_name)
            shutil.copy(src_mask, output_dataset_dir / "masks" / output_name)
            print(f"  Copied: {output_name}")
        else:
            print(f"  WARNING: Missing files for {image_name}")
    
    return selected_images


def create_metadata_file(output_path, pannuke_imgs, tcga_imgs, siim_imgs):
    """Create a metadata CSV file summarizing the selected images."""
    metadata = []
    
    # Pannuke metadata
    for i, img in enumerate(pannuke_imgs):
        metadata.append({
            'dataset': 'pannuke',
            'index': i,
            'image_id': img['image_name'],
            'label': img['label'],
            'processing': 'macenko',
            'group_id': img['group_id']
        })
    
    # TCGA metadata
    for i, img in enumerate(tcga_imgs):
        metadata.append({
            'dataset': 'tcga',
            'index': i,
            'image_id': img['image_name'],
            'label': img['patient'],
            'processing': 'raw',
            'group_id': img['patient']
        })
    
    # SIIM metadata
    for i, img in enumerate(siim_imgs):
        metadata.append({
            'dataset': 'siim',
            'index': i,
            'image_id': img['image_id'],
            'label': img['label'],
            'processing': 'pneumothorax_classification',
            'group_id': img['image_id']
        })
    
    metadata_df = pd.DataFrame(metadata)
    metadata_path = output_path / "METADATA.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\nMetadata saved to: {metadata_path}")
    print(f"\nTotal images selected: {len(metadata)} (10 from each dataset)")
    return metadata_df


def main():
    base_path = "/home/apath/Work/temp/final"
    
    # Create output structure
    output_path = create_output_directory(base_path)
    print(f"\nOutput directory created: {output_path}")
    
    # Select images from each dataset
    pannuke_imgs = select_pannuke_images(base_path, output_path, num_images=10)
    tcga_imgs = select_tcga_images(base_path, output_path, num_images=10)
    siim_imgs = select_siim_images(base_path, output_path, num_images=10)
    
    # Create metadata
    create_metadata_file(output_path, pannuke_imgs, tcga_imgs, siim_imgs)
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Output location: {output_path}")
    print(f"Structure:")
    print(f"  - pannuke/images/  (10 macenko processed images)")
    print(f"  - pannuke/masks/")
    print(f"  - tcga/images/     (10 images from different patients)")
    print(f"  - tcga/masks/")
    print(f"  - siim/images/     (10 pneumothorax classification images)")
    print(f"  - siim/masks/")
    print(f"  - METADATA.csv     (comprehensive index)")
    print("\nAll images are paired with their corresponding masks and labeled with class information.")
    print("="*60)


if __name__ == "__main__":
    main()
