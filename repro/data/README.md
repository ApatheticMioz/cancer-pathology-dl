# Multi-Dataset Image Subset with Paired Masks and Labels

This directory contains a structured subset of images from three medical imaging datasets, each containing 10 images paired with their segmentation masks and classification labels.

## Directory Structure

```
subset_10_images/
├── pannuke/
│   ├── images/          (10 macenko-processed histopathology images)
│   └── masks/           (corresponding segmentation masks)
├── tcga/
│   ├── images/          (10 brain tumor MRI images from different patients)
│   └── masks/           (corresponding segmentation masks)
├── siim/
│   ├── images/          (10 chest X-ray images)
│   └── masks/           (corresponding pneumothorax segmentation masks)
├── METADATA.csv         (comprehensive index file)
└── README.md           (this file)
```

## Dataset Details

### PANNUKE Dataset
**Type**: Histopathology image dataset (tissue segmentation)
**Number of images**: 10
**Image format**: PNG (macenko-normalized color-adjusted images)
**Mask format**: PNG (segmentation masks)
**Labels**: Tissue type classification (10 different labels: 0-9)

- Label 0, 1, 2, ... 9 represent different tissue and nuclei types
- Each image is paired with a binary mask identifying regions of interest
- Macenko preprocessing: color normalization to reduce staining variations
- Original resolution: Various (preprocessed to consistent size)

**Image naming convention**: `{index:02d}_label_{label}_{original_name}.png`

**Example files**:
- `00_label_0_fold_1_01171.png` (image index 0, tissue type 0)
- `09_label_9_fold_1_02189.png` (image index 9, tissue type 9)

### TCGA Dataset
**Type**: Brain tumor MRI imaging dataset
**Number of images**: 10
**Image format**: TIFF (16-bit grayscale)
**Mask format**: TIFF (binary segmentation masks)
**Labels**: Patient identifiers for grouping images by source

- Images are selected from 10 different patients (tumor diversity)
- Each patient contributes 1 image to ensure maximum diversity
- TCGA standard ID format: `TCGA_{disease_code}_{patient_code}_{date}`
- Disease codes in this subset: CS (colorectal squamous cell carcinoma), DU, FG, HT, EZ

**Image naming convention**: `{index:02d}_{patient_short_id}_{original_name}.tif`

**Example files**:
- `00_CS_4941_19960909_TCGA_CS_4941_19960909_16.tif` (patient 1, image 1)
- `09_CS_5393_19990606_TCGA_CS_5393_19990606_5.tif` (patient 10, image 2)

### SIIM Dataset
**Type**: Chest X-ray dataset (pneumothorax detection)
**Number of images**: 10
**Image format**: PNG
**Mask format**: PNG (binary segmentation masks)
**Labels**: Binary classification (0 = normal, 1 = pneumothorax present)
**Label distribution**: 6 normal, 4 with pneumothorax

- Images are chest X-rays from the SIIM-ACR Pneumothorax Segmentation Challenge
- Masks indicate regions where pneumothorax is present
- Task: Detect and segment collapsed lung areas

**Image naming convention**: `{index:02d}_label_{label}_{label_text}_{original_id}.png`

**Example files**:
- `00_label_0_normal_1.2.276.0.7230010.3.1.4.8323329.1724.1517875169.115602.png` (normal)
- `06_label_1_pneumothorax_1.2.276.0.7230010.3.1.4.8323329.13920.1517875248.649448.png` (pneumothorax)

## METADATA.csv File Format

The METADATA.csv file contains:

| Column | Description |
|--------|-------------|
| `dataset` | Source dataset (pannuke, tcga, or siim) |
| `index` | Image index within the dataset (0-9) |
| `image_id` | Original image identifier |
| `label` | Classification label or tissue type |
| `processing` | Image preprocessing applied (macenko, raw, pneumothorax_classification) |
| `group_id` | Grouping identifier (fold ID for pannuke, patient ID for tcga, image ID for siim) |

### Sample METADATA.csv Structure

```csv
dataset,index,image_id,label,processing,group_id
pannuke,0,fold_1_01171.png,0,macenko,fold_1_01171
pannuke,1,fold_1_01370.png,1,macenko,fold_1_01370
...
tcga,0,TCGA_CS_4941_19960909_16.tif,TCGA_CS_4941_19960909,raw,TCGA_CS_4941_19960909
...
siim,0,1.2.276.0.7230010.3.1.4.8323329.1724.1517875169.115602.png,0,pneumothorax_classification,1.2.276.0.7230010.3.1.4.8323329.1724.1517875169.115602
```

## Label Diversity

### PANNUKE (Tissue Classification)
✅ **Maximum diversity achieved**: 10 different tissue types from 10 unique images
- Labels: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
- Each image represents a distinct tissue or nuclei classification

### TCGA (Patient Diversity)
✅ **Maximum diversity achieved**: 10 images from 10 different patients
- Patient-level diversity ensures distinct tumor characteristics
- Cross-patient variability in pathology and imaging protocols

### SIIM (Pneumothorax Detection)
✅ **Balanced binary classification**: 6 normal, 4 pneumothorax cases
- Sufficient examples of both classes for model evaluation

## Key Features

1. **Paired Data**: Every image has a corresponding segmentation mask
2. **Labeled**: All images include classification labels or identifiers
3. **Diverse**: Images are selected to maximize label diversity:
   - Pannuke: All 10 images have different tissue type labels
   - TCGA: All 10 images from different patients
   - SIIM: Balanced representation of both classes
4. **Structured**: Organized into dataset-specific subdirectories
5. **Documented**: Complete metadata in CSV format for easy integration

## Usage Examples

### Loading with Pandas
```python
import pandas as pd
import cv2
from pathlib import Path

# Load metadata
metadata = pd.read_csv('subset_10_images/METADATA.csv')

# Get pannuke images
pannuke_images = metadata[metadata['dataset'] == 'pannuke']
for idx, row in pannuke_images.iterrows():
    img = cv2.imread(f"subset_10_images/pannuke/images/{row['image_id']}")
    mask = cv2.imread(f"subset_10_images/pannuke/masks/{row['image_id']}", cv2.IMREAD_GRAYSCALE)
    label = row['label']
    print(f"Tissue type {label}: {img.shape}")
```

### Loading with PyTorch
```python
from torch.utils.data import Dataset
from torchvision import transforms
import cv2

class MedicalImageDataset(Dataset):
    def __init__(self, metadata_csv, root_dir):
        self.metadata = pd.read_csv(metadata_csv)
        self.root_dir = Path(root_dir)
        self.transform = transforms.ToTensor()
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        dataset = row['dataset']
        
        # Load image and mask
        img_path = self.root_dir / dataset / 'images' / row['image_id']
        mask_path = self.root_dir / dataset / 'masks' / row['image_id']
        
        image = cv2.imread(str(img_path))
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask, row['label']

# Usage
dataset = MedicalImageDataset('subset_10_images/METADATA.csv', 'subset_10_images')
```

## Statistics

| Dataset | Count | Image Format | Mask Format | Labels | Diversity |
|---------|-------|--------------|------------|--------|-----------|
| PANNUKE | 10 | PNG | PNG | 0-9 (10 unique) | ✅ All different |
| TCGA | 10 | TIFF | TIFF | Patient IDs | ✅ All different |
| SIIM | 10 | PNG | PNG | 0-1 (binary) | ✅ 6 normal, 4 pneumothorax |
| **Total** | **30** | Mixed | Mixed | Mixed | **✅ Maximized** |

## Original Data Locations

The complete datasets are available in:
- `pannuke/preprocessed_macenko/` - Full pannuke macenko-processed dataset
- `TCGA/` - Complete TCGA brain tumor dataset
- `SIIM_raw/preprocessed/` - Full SIIM pneumothorax dataset

## Creation Notes

- **Created**: 2026-05-07
- **Selection method**: Stratified sampling to maximize label diversity
- **Processing**: Macenko normalization applied to pannuke images
- **Purpose**: Provide a representative, balanced subset for multi-dataset model evaluation

## File Sizes

Typical file sizes:
- Pannuke images: 256x256 PNG (~100-150 KB)
- TCGA images: Variable resolution TIFF (~500 KB - 2 MB)
- SIIM images: 1024x1024 PNG (~300-500 KB)

**Total subset size**: ~50-60 MB (depending on compression)

## License and Attribution

This subset maintains the original licenses of the source datasets:
- **PANNUKE**: Public dataset from research publication
- **TCGA**: Public cancer imaging data (National Cancer Institute)
- **SIIM**: Pneumothorax challenge dataset with research license

Please cite the original datasets when using this subset in publications.
