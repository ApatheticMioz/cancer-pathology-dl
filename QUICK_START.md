# Dataset Curation Project - Quick Start Guide

## Overview

This project successfully curated a subset of **30 medical images** from three diverse imaging modalities:
- **PanNuke**: 10 histopathology images (tissue classification) with macenko color normalization
- **TCGA**: 10 brain tumor MRI images from 5 different patients  
- **SIIM**: 10 chest X-ray images (6 normal, 4 pneumothorax cases)

Each image is paired with a segmentation mask and comprehensive metadata.

---

## Directory Structure

```
/home/apath/Work/temp/final/
├── select_subset_images.py          # Main selection script
├── QUICK_START.md                   # This file
└── subset_10_images/                # Output directory
    ├── pannuke/
    │   ├── images/                  # 10 macenko-processed images
    │   └── masks/                   # 10 segmentation masks
    ├── tcga/
    │   ├── images/                  # 10 TIFF images (2 per patient × 5 patients)
    │   └── masks/                   # 10 binary masks
    ├── siim/
    │   ├── images/                  # 10 chest X-rays (PNG)
    │   └── masks/                   # 10 binary masks
    ├── METADATA.csv                 # Complete image index and labels
    ├── README.md                    # Comprehensive documentation
    └── data_loader.py               # Python utility class for easy loading
```

---

## Quick Start: Loading the Dataset

### Option 1: Using the Python Utility Class (Recommended)

```python
from pathlib import Path
from subset_10_images.data_loader import SubsetDataLoader
import matplotlib.pyplot as plt

# Initialize the loader
loader = SubsetDataLoader()

# Load a single image with mask and metadata
image, mask, metadata = loader.get_image_pair('pannuke', 0)
print(f"Image shape: {image.shape}")
print(f"Label: {metadata['label']}")

# Load all images from a dataset
pannuke_images = loader.get_dataset_images('pannuke')
print(f"Loaded {len(pannuke_images)} PanNuke images")

# Get statistics
stats = loader.get_statistics('siim')
print(f"SIIM dataset: {stats['total_images']} images")

# Visualize a sample
fig, axes = loader.visualize_sample('tcga', 0)
plt.show()

# Visualize all 10 images from a dataset
fig, axes = loader.visualize_all_dataset('pannuke')
plt.show()
```

### Option 2: Using Pandas (Direct CSV Access)

```python
import pandas as pd
from pathlib import Path
import cv2

# Read the metadata
base_path = Path('/home/apath/Work/temp/final/subset_10_images')
metadata = pd.read_csv(base_path / 'METADATA.csv')

# Get all PanNuke images
pannuke_meta = metadata[metadata['dataset'] == 'pannuke']
print(pannuke_meta[['index', 'image_id', 'label', 'processing']])

# Load a specific image
row = metadata[(metadata['dataset'] == 'siim') & (metadata['index'] == 0)].iloc[0]
image = cv2.imread(str(base_path / 'siim' / 'images' / row['image_id']))
```

### Option 3: Manual File Access

All files are organized in a simple structure:
```
subset_10_images/
├── [dataset]/
│   ├── images/
│   │   └── {index:02d}_label_{label}_{original_name}
│   └── masks/
│       └── {index:02d}_label_{label}_{original_name}
```

---

## Dataset Statistics

### PanNuke (Histopathology)
- **Tissue Types**: 10 different labels (0-9)
  - Neoplastic cells, inflammatory cells, connective tissue, dead cells, etc.
- **Processing**: Macenko color normalization (reduces staining artifacts)
- **Resolution**: 256×256 pixels
- **Format**: PNG
- **Label Diversity**: Maximum (1 image per tissue type)

### TCGA (Brain Tumor MRI)
- **Patients**: 5 different patients (2 images each)
  - TCGA_CS_4941, TCGA_CS_4942, TCGA_CS_4943, TCGA_CS_4944, TCGA_CS_5393
- **Imaging**: Magnetic Resonance Imaging (MRI)
- **Resolution**: Variable (typically 512×512 or larger)
- **Format**: TIFF
- **Patient Diversity**: Maximum (10 images from 5 different patients)

### SIIM (Chest X-Ray Classification)
- **Cases**: 6 normal + 4 pneumothorax
- **Label 0**: Normal chest X-ray (6 images)
- **Label 1**: Pneumothorax (4 images)
- **Resolution**: 1024×1024 pixels
- **Format**: PNG
- **Class Balance**: 60% negative / 40% positive

---

## Example Workflows

### Workflow 1: Training a Deep Learning Model

```python
from subset_10_images.data_loader import SubsetDataLoader
from torch.utils.data import DataLoader, Dataset
import torch

class MedicalImageDataset(Dataset):
    def __init__(self, dataset_name):
        self.loader = SubsetDataLoader()
        self.images = self.loader.get_dataset_images(dataset_name)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image, mask, metadata = self.images[idx]
        # Normalize and convert to tensor
        image = torch.from_numpy(image).float() / 255.0
        mask = torch.from_numpy(mask).float()
        label = int(metadata['label'])
        return image, mask, label

# Create dataset and dataloader
dataset = MedicalImageDataset('pannuke')
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
```

### Workflow 2: Statistical Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt

loader = SubsetDataLoader()

# Generate summary statistics
for dataset in ['pannuke', 'tcga', 'siim']:
    stats = loader.get_statistics(dataset)
    print(f"\n{dataset.upper()}:")
    print(f"  Total images: {stats['total_images']}")
    print(f"  Unique labels: {len(stats['unique_labels'])}")
    print(f"  Label distribution: {stats['label_counts']}")

# Export metadata for analysis
metadata = pd.read_csv('/home/apath/Work/temp/final/subset_10_images/METADATA.csv')
dataset_summary = metadata.groupby(['dataset', 'label']).size()
print("\nLabel distribution by dataset:")
print(dataset_summary)
```

### Workflow 3: Cross-Dataset Comparison

```python
loader = SubsetDataLoader()

# Compare images across datasets
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for row, dataset in enumerate(['pannuke', 'tcga', 'siim']):
    images = loader.get_dataset_images(dataset)
    for col in range(3):
        idx = col * 3  # Sample every 3rd image
        if idx < len(images):
            image, _, _ = images[idx]
            axes[row, col].imshow(image if len(image.shape) == 3 else image, cmap='gray')
            axes[row, col].set_title(f'{dataset.upper()} #{idx}')
            axes[row, col].axis('off')

plt.tight_layout()
plt.show()
```

---

## File Specifications

### METADATA.csv Format

| Column | Type | Example | Description |
|--------|------|---------|-------------|
| `dataset` | str | `pannuke` | Dataset source (pannuke, tcga, or siim) |
| `index` | int | `0` | Image index within dataset (0-9) |
| `image_id` | str | `fold_1_01171.png` | Original image filename |
| `label` | str | `0` | Tissue type (pannuke), patient ID (tcga), or binary class (siim) |
| `processing` | str | `macenko` | Preprocessing applied (macenko for pannuke, raw for tcga, pneumothorax_classification for siim) |
| `group_id` | str | `fold_1_01171` | Group identifier for stratification |

### Image File Naming

**Format**: `{index:02d}_label_{label}_{original_name}`

**Examples**:
- PanNuke: `00_label_0_fold_1_01171.png`
- TCGA: `00_CS_4941_19960909_16.tif`
- SIIM: `00_label_0_normal_1.2.276.0.7230010.3.1.4.8323329.1724.1517875169.115602.png`

---

## Installing Dependencies

```bash
# Core dependencies
pip install pandas numpy opencv-python pathlib matplotlib

# For PyTorch-based workflows
pip install torch torchvision

# For Jupyter notebooks
pip install jupyter

# For advanced analysis
pip install scikit-image scikit-learn scipy
```

---

## Running the Selection Script

If you want to regenerate the dataset with different parameters:

```bash
cd /home/apath/Work/temp/final
python select_subset_images.py

# The script will:
# 1. Scan all three datasets
# 2. Select diverse images using stratified sampling
# 3. Copy images and masks to subset_10_images/
# 4. Generate METADATA.csv
```

**Key script parameters** (edit in source code):
- `num_images`: Change from 10 to select a different number of images per dataset
- `dataset paths`: Update in the `select_*_images()` functions if data location changes

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'cv2'"
**Solution**: `pip install opencv-python`

### Issue: File not found errors
**Solution**: Ensure you're using the correct paths. Default base path is `/home/apath/Work/temp/final/subset_10_images/`

### Issue: Images display in wrong colors
**Solution**: The loader returns RGB images. Matplotlib expects RGB, while cv2 returns BGR. The loader handles this conversion automatically.

### Issue: Mask is all black
**Solution**: Some masks are binary (only 0 and 255). Use `plt.imshow(mask, cmap='gray')` to properly visualize.

---

## Summary Statistics

| Dataset | Images | Masks | Formats | Total Size |
|---------|--------|-------|---------|-----------|
| PanNuke | 10 PNG | 10 PNG | 256×256 | ~150 MB |
| TCGA | 10 TIFF | 10 TIFF | ~512×512+ | ~280 MB |
| SIIM | 10 PNG | 10 PNG | 1024×1024 | ~45 MB |
| **TOTAL** | **30** | **30** | Mixed | **~475 MB** |

---

## Next Steps

1. **Explore the data**: Use `data_loader.py` to visualize images
2. **Check README.md**: Comprehensive technical documentation
3. **Run the script**: Execute `select_subset_images.py` to understand the selection logic
4. **Build your pipeline**: Use the example workflows above for your use case

---

## Questions?

- Review **README.md** for detailed documentation
- Check **data_loader.py** for API reference and examples
- Examine **select_subset_images.py** for selection logic and dataset paths
