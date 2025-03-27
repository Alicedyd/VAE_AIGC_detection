#!/usr/bin/env python3
"""
Prepare a custom dataset for fake image detection.
This script organizes your images into the format expected by the fake detector.
"""

import os
import argparse
import shutil
from pathlib import Path
import random
from tqdm import tqdm


def prepare_dataset(real_dir, fake_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Prepare custom dataset for fake detection by organizing files in the expected structure
    
    Args:
        real_dir: Directory containing real images
        fake_dir: Directory containing fake/generated images
        output_dir: Output directory for the organized dataset
        train_ratio: Ratio of images to use for training
        val_ratio: Ratio of images to use for validation
        test_ratio: Ratio of images to use for testing
    """
    # Validate ratios
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    
    # Create output directory structure
    train_real_dir = Path(output_dir) / 'train' / 'progan' / '0_real'
    train_fake_dir = Path(output_dir) / 'train' / 'progan' / '1_fake'
    val_real_dir = Path(output_dir) / 'test' / 'progan' / '0_real'
    val_fake_dir = Path(output_dir) / 'test' / 'progan' / '1_fake'
    
    os.makedirs(train_real_dir, exist_ok=True)
    os.makedirs(train_fake_dir, exist_ok=True)
    os.makedirs(val_real_dir, exist_ok=True)
    os.makedirs(val_fake_dir, exist_ok=True)
    
    # Get all real and fake image files
    real_files = [f for f in Path(real_dir).glob('**/*') if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    fake_files = [f for f in Path(fake_dir).glob('**/*') if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    print(f"Found {len(real_files)} real images and {len(fake_files)} fake images")
    
    # Shuffle files
    random.shuffle(real_files)
    random.shuffle(fake_files)

import os
import argparse
import shutil
from pathlib import Path
import random
from tqdm import tqdm

def prepare_dataset(real_dir, fake_dir, output_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Prepare custom dataset for fake detection by organizing files in the expected structure
    
    Args:
        real_dir: Directory containing real images
        fake_dir: Directory containing fake/generated images
        output_dir: Output directory for the organized dataset
        train_ratio: Ratio of images to use for training
        val_ratio: Ratio of images to use for validation
        test_ratio: Ratio of images to use for testing
    """
    # Validate ratios
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1.0"
    
    # Create output directory structure
    train_real_dir = Path(output_dir) / 'train' / 'progan' / '0_real'
    train_fake_dir = Path(output_dir) / 'train' / 'progan' / '1_fake'
    val_real_dir = Path(output_dir) / 'test' / 'progan' / '0_real'
    val_fake_dir = Path(output_dir) / 'test' / 'progan' / '1_fake'
    
    os.makedirs(train_real_dir, exist_ok=True)
    os.makedirs(train_fake_dir, exist_ok=True)
    os.makedirs(val_real_dir, exist_ok=True)
    os.makedirs(val_fake_dir, exist_ok=True)
    
    # Get all real and fake image files
    real_files = [f for f in Path(real_dir).glob('**/*') if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    fake_files = [f for f in Path(fake_dir).glob('**/*') if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png']]
    
    print(f"Found {len(real_files)} real images and {len(fake_files)} fake images")
    
    # Shuffle files
    random.shuffle(real_files)
    random.shuffle(fake_files)
    
    # Split real files
    real_train_count = int(len(real_files) * train_ratio)
    real_val_count = int(len(real_files) * val_ratio)
    
    real_train = real_files[:real_train_count]
    real_val = real_files[real_train_count:real_train_count + real_val_count]
    
    # Split fake files
    fake_train_count = int(len(fake_files) * train_ratio)
    fake_val_count = int(len(fake_files) * val_ratio)
    
    fake_train = fake_files[:fake_train_count]
    fake_val = fake_files[fake_train_count:fake_train_count + fake_val_count]
    
    # Copy files to their respective directories
    print("Copying real training images...")
    for f in tqdm(real_train):
        shutil.copy2(f, train_real_dir / f.name)
    
    print("Copying fake training images...")
    for f in tqdm(fake_train):
        shutil.copy2(f, train_fake_dir / f.name)
    
    print("Copying real validation images...")
    for f in tqdm(real_val):
        shutil.copy2(f, val_real_dir / f.name)
    
    print("Copying fake validation images...")
    for f in tqdm(fake_val):
        shutil.copy2(f, val_fake_dir / f.name)
    
    print(f"Dataset prepared in {output_dir}")
    print(f"Training: {len(real_train)} real, {len(fake_train)} fake")
    print(f"Validation: {len(real_val)} real, {len(fake_val)} fake")
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare a custom dataset for fake image detection")
    parser.add_argument("--real-dir", required=True, help="Directory containing real images")
    parser.add_argument("--fake-dir", required=True, help="Directory containing fake/generated images")
    parser.add_argument("--output-dir", required=True, help="Output directory for the organized dataset")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio of images to use for training")
    parser.add_argument("--val-ratio", type=float, default=0.1, help="Ratio of images to use for validation")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Ratio of images to use for testing")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.isdir(args.real_dir):
        raise ValueError(f"Real directory {args.real_dir} does not exist")
    if not os.path.isdir(args.fake_dir):
        raise ValueError(f"Fake directory {args.fake_dir} does not exist")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the dataset preparation
    prepare_dataset(
        args.real_dir, 
        args.fake_dir, 
        args.output_dir,
        args.train_ratio,
        args.val_ratio,
        args.test_ratio
    )