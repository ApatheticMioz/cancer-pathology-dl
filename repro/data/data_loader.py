#!/usr/bin/env python3
"""
Utility functions for loading and working with the subset_10_images dataset.
This script provides convenient functions for accessing images, masks, and labels.
"""

import pandas as pd
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Dict, List, Optional
import matplotlib.pyplot as plt


class SubsetDataLoader:
    """
    Convenient loader for the subset_10_images dataset with all three datasets.
    """
    
    def __init__(self, root_path: str = "/home/apath/Work/temp/final/subset_10_images"):
        """
        Initialize the data loader.
        
        Args:
            root_path: Path to the subset_10_images directory
        """
        self.root_path = Path(root_path)
        self.metadata = pd.read_csv(self.root_path / "METADATA.csv")
        
    def load_image(self, dataset: str, index: int, grayscale: bool = False) -> np.ndarray:
        """
        Load an image from the specified dataset.
        
        Args:
            dataset: Dataset name ('pannuke', 'tcga', or 'siim')
            index: Image index within the dataset (0-9)
            grayscale: If True, convert to grayscale
            
        Returns:
            Image array (HxWxC or HxW if grayscale)
        """
        row = self.metadata[(self.metadata['dataset'] == dataset) & 
                           (self.metadata['index'] == index)].iloc[0]
        
        # Different naming schemes for different datasets
        if dataset == 'pannuke':
            # Format: {index:02d}_label_{label}_{original_name}
            original_name = row['image_id']
            label = row['label']
            prefixed_name = f"{index:02d}_label_{label}_{original_name}"
        elif dataset == 'tcga':
            # Format: {index:02d}_{patient_short}_{image_name}
            # We can just list files and match by index
            img_dir = self.root_path / dataset / 'images'
            all_files = sorted([f for f in img_dir.glob('*') if f.is_file()])
            if index < len(all_files):
                prefixed_name = all_files[index].name
            else:
                raise IndexError(f"Image index {index} out of range for {dataset}")
        elif dataset == 'siim':
            # Format: {index:02d}_label_{label}_{label_text}_{image_id}.png
            img_dir = self.root_path / dataset / 'images'
            all_files = sorted([f for f in img_dir.glob('*') if f.is_file()])
            if index < len(all_files):
                prefixed_name = all_files[index].name
            else:
                raise IndexError(f"Image index {index} out of range for {dataset}")
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        img_path = self.root_path / dataset / 'images' / prefixed_name
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError(f"Failed to load image: {img_path}")
        
        if grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    def load_mask(self, dataset: str, index: int) -> np.ndarray:
        """
        Load a segmentation mask from the specified dataset.
        
        Args:
            dataset: Dataset name ('pannuke', 'tcga', or 'siim')
            index: Image index within the dataset (0-9)
            
        Returns:
            Mask array (HxW, binary or multi-class)
        """
        row = self.metadata[(self.metadata['dataset'] == dataset) & 
                           (self.metadata['index'] == index)].iloc[0]
        
        # Different naming schemes for different datasets (same as images)
        if dataset == 'pannuke':
            # Format: {index:02d}_label_{label}_{original_name}
            original_name = row['image_id']
            label = row['label']
            prefixed_name = f"{index:02d}_label_{label}_{original_name}"
        elif dataset == 'tcga':
            # Format: {index:02d}_{patient_short}_{image_name}
            # We can just list files and match by index
            mask_dir = self.root_path / dataset / 'masks'
            all_files = sorted([f for f in mask_dir.glob('*') if f.is_file()])
            if index < len(all_files):
                prefixed_name = all_files[index].name
            else:
                raise IndexError(f"Mask index {index} out of range for {dataset}")
        elif dataset == 'siim':
            # Format: {index:02d}_label_{label}_{label_text}_{image_id}.png
            mask_dir = self.root_path / dataset / 'masks'
            all_files = sorted([f for f in mask_dir.glob('*') if f.is_file()])
            if index < len(all_files):
                prefixed_name = all_files[index].name
            else:
                raise IndexError(f"Mask index {index} out of range for {dataset}")
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        mask_path = self.root_path / dataset / 'masks' / prefixed_name
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Failed to load mask: {mask_path}")
        
        return mask
    
    def get_image_pair(self, dataset: str, index: int) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Load both image and mask together with metadata.
        
        Args:
            dataset: Dataset name ('pannuke', 'tcga', or 'siim')
            index: Image index within the dataset (0-9)
            
        Returns:
            Tuple of (image, mask, metadata_dict)
        """
        image = self.load_image(dataset, index)
        mask = self.load_mask(dataset, index)
        row = self.metadata[(self.metadata['dataset'] == dataset) & 
                           (self.metadata['index'] == index)].iloc[0]
        metadata = row.to_dict()
        
        return image, mask, metadata
    
    def get_dataset_images(self, dataset: str) -> List[Tuple[np.ndarray, np.ndarray, Dict]]:
        """
        Load all 10 images and masks from a specific dataset.
        
        Args:
            dataset: Dataset name ('pannuke', 'tcga', or 'siim')
            
        Returns:
            List of (image, mask, metadata) tuples
        """
        images = []
        dataset_rows = self.metadata[self.metadata['dataset'] == dataset]
        
        for _, row in dataset_rows.iterrows():
            image, mask, meta = self.get_image_pair(dataset, row['index'])
            images.append((image, mask, meta))
        
        return images
    
    def get_statistics(self, dataset: Optional[str] = None) -> Dict:
        """
        Get statistics about the dataset.
        
        Args:
            dataset: If specified, return stats for only this dataset.
                    Otherwise, return stats for all datasets.
            
        Returns:
            Dictionary of statistics
        """
        if dataset:
            subset = self.metadata[self.metadata['dataset'] == dataset]
        else:
            subset = self.metadata
        
        stats = {
            'total_images': len(subset),
            'datasets': subset['dataset'].unique().tolist() if not dataset else [dataset],
            'unique_labels': subset['label'].unique().tolist(),
            'label_counts': subset['label'].value_counts().to_dict()
        }
        
        return stats
    
    def visualize_sample(self, dataset: str, index: int, figsize: Tuple[int, int] = (15, 5)):
        """
        Visualize an image with its mask.
        
        Args:
            dataset: Dataset name ('pannuke', 'tcga', or 'siim')
            index: Image index within the dataset (0-9)
            figsize: Figure size for matplotlib
        """
        image, mask, metadata = self.get_image_pair(dataset, index)
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Original image
        if len(image.shape) == 3:
            axes[0].imshow(image)
        else:
            axes[0].imshow(image, cmap='gray')
        axes[0].set_title(f'{dataset.upper()} - Image\n(Label: {metadata["label"]})')
        axes[0].axis('off')
        
        # Mask
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title(f'{dataset.upper()} - Mask')
        axes[1].axis('off')
        
        # Overlay
        if len(image.shape) == 3:
            overlay = image.copy().astype(float)
            overlay[mask > 0] = (overlay[mask > 0] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
        else:
            overlay = image.copy()
        
        if len(image.shape) == 3:
            axes[2].imshow(overlay.astype(np.uint8))
        else:
            axes[2].imshow(overlay, cmap='gray')
        axes[2].set_title(f'{dataset.upper()} - Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        return fig, axes
    
    def visualize_all_dataset(self, dataset: str, figsize: Tuple[int, int] = (20, 15)):
        """
        Visualize all 10 images from a dataset in a grid.
        
        Args:
            dataset: Dataset name ('pannuke', 'tcga', or 'siim')
            figsize: Figure size for matplotlib
        """
        images = self.get_dataset_images(dataset)
        
        fig, axes = plt.subplots(10, 2, figsize=figsize)
        
        for i, (image, mask, metadata) in enumerate(images):
            # Image column
            if len(image.shape) == 3:
                axes[i, 0].imshow(image)
            else:
                axes[i, 0].imshow(image, cmap='gray')
            axes[i, 0].set_title(f'Image {i} - Label: {metadata["label"]}')
            axes[i, 0].axis('off')
            
            # Mask column
            axes[i, 1].imshow(mask, cmap='gray')
            axes[i, 1].set_title(f'Mask {i}')
            axes[i, 1].axis('off')
        
        plt.suptitle(f'{dataset.upper()} Dataset - All Images and Masks', fontsize=16, y=0.995)
        plt.tight_layout()
        return fig, axes


def print_dataset_info():
    """Print detailed information about the subset dataset."""
    loader = SubsetDataLoader()
    
    print("="*70)
    print("SUBSET_10_IMAGES DATASET SUMMARY")
    print("="*70)
    
    for dataset in ['pannuke', 'tcga', 'siim']:
        stats = loader.get_statistics(dataset)
        print(f"\n{dataset.upper()}:")
        print(f"  Total images: {stats['total_images']}")
        print(f"  Unique labels: {len(stats['unique_labels'])}")
        print(f"  Labels: {stats['unique_labels'][:10]}{'...' if len(stats['unique_labels']) > 10 else ''}")
        print(f"  Label distribution:")
        for label, count in sorted(stats['label_counts'].items()):
            print(f"    Label {label}: {count} images")
    
    print("\n" + "="*70)
    print("TOTAL: 30 images (10 from each dataset) with paired masks and labels")
    print("="*70)


def example_usage():
    """Example usage of the SubsetDataLoader."""
    
    # Initialize loader
    loader = SubsetDataLoader()
    
    print("\n" + "="*70)
    print("EXAMPLE: Loading and Working with the Dataset")
    print("="*70)
    
    # Load a single image pair
    print("\n1. Load a single image pair:")
    image, mask, metadata = loader.get_image_pair('pannuke', 0)
    print(f"   Image shape: {image.shape}")
    print(f"   Mask shape: {mask.shape}")
    print(f"   Metadata: {metadata}")
    
    # Load all images from a dataset
    print("\n2. Load all images from SIIM dataset:")
    siim_images = loader.get_dataset_images('siim')
    print(f"   Loaded {len(siim_images)} images")
    for i, (img, mask, meta) in enumerate(siim_images[:3]):
        print(f"     Image {i}: {img.shape}, Label: {meta['label']}")
    
    # Get statistics
    print("\n3. Get dataset statistics:")
    stats = loader.get_statistics()
    print(f"   Total images across all datasets: {stats['total_images']}")
    print(f"   Datasets: {stats['datasets']}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    # Print dataset information
    print_dataset_info()
    
    # Show example usage
    example_usage()
    
    # Optional: Visualize samples
    print("\nTo visualize samples, use:")
    print("  loader = SubsetDataLoader()")
    print("  fig, axes = loader.visualize_sample('pannuke', 0)")
    print("  plt.show()")
    print("\nOr visualize all images from a dataset:")
    print("  fig, axes = loader.visualize_all_dataset('siim')")
    print("  plt.show()")
