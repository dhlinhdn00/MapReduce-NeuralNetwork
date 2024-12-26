#!/usr/bin/env python3
import os
import sys
import numpy as np
from PIL import Image, ImageOps, ImageChops
import argparse
import random
from PIL import ImageEnhance

def preprocess(input_dir, output_dir, percent, augment):
    subsets = ['train', 'test']
    
    for subset in subsets:
        subset_dir = os.path.join(input_dir, subset)
        output_file = os.path.join(output_dir, f"mnist_{subset}.txt")
        
        with open(output_file, 'w') as out_f:
            for label in map(str, range(10)):
                label_dir = os.path.join(subset_dir, label)
                
                if not os.path.isdir(label_dir):
                    print(f"Directory {label_dir} does not exist. Skipping.")
                    continue
                
                files = [file for file in os.listdir(label_dir) if file.endswith('.png')]
                
                if not files:
                    print(f"No PNG files in directory {label_dir}. Skipping.")
                    continue
                
                total_files = len(files)
                num_selected = int(total_files * percent / 100)
                
                if num_selected == 0:
                    print(f"Percentage {percent}% is too low for directory {label_dir}. Skipping.")
                    continue
                
                random.seed(42)
                selected_files = random.sample(files, min(num_selected, total_files))
                
                for file in selected_files:
                    path = os.path.join(label_dir, file)
                    try:
                        image = Image.open(path).convert('L')  # Convert to grayscale
                        image = image.resize((28, 28))  # Resize image to 28x28 pixels
                        pixels = np.array(image).flatten() / 255.0  # Normalize pixel values
                        label_int = int(label)  # Convert label to integer
                        pixels_str = ' '.join(map(str, pixels))  # Convert pixel values to string
                        out_f.write(f"{label_int} {pixels_str}\n")  # Write label and pixel values to output file
                        
                        if augment:
                            augmented_images = augment_image(image)
                            for aug_image in augmented_images:
                                aug_pixels = np.array(aug_image).flatten() / 255.0
                                aug_pixels_str = ' '.join(map(str, aug_pixels))
                                out_f.write(f"{label_int} {aug_pixels_str}\n")
                    except Exception as e:
                        print(f"Error processing {path}: {e}", file=sys.stderr)

def augment_image(image):
    """
    Apply a variety of random augmentations to the input image.
    Returns a list of augmented images.
    """
    augmented = []
    
    # Define the number of augmentations per image
    num_augmentations = 3
    
    for _ in range(num_augmentations):
        aug_image = image.copy()
        
        # Apply random rotation between -20 to 20 degrees
        angle = random.uniform(-20, 20)
        aug_image = aug_image.rotate(angle, fillcolor=0)
        
        # Apply random shift
        max_shift = 3  # pixels
        shift_x = random.randint(-max_shift, max_shift)
        shift_y = random.randint(-max_shift, max_shift)
        aug_image = ImageChops.offset(aug_image, shift_x, shift_y)
        
        # Apply random scaling
        scale_factor = random.uniform(0.9, 1.1)
        new_size = (int(28 * scale_factor), int(28 * scale_factor))
        aug_image = aug_image.resize(new_size, Image.LANCZOS)
        aug_image = ImageOps.fit(aug_image, (28, 28), Image.LANCZOS)
        
        # Apply random shearing
        shear_factor = random.uniform(-10, 10)
        aug_image = aug_image.transform(
            aug_image.size,
            Image.AFFINE,
            (1, shear_factor / 100, 0, 0, 1, 0),
            Image.BICUBIC,
            fillcolor=0
        )
        
        # Apply elastic distortion
        aug_image = apply_elastic_distortion(aug_image)

        # Add Gaussian noise
        aug_image = add_gaussian_noise(aug_image)
        
        augmented.append(aug_image)
    
    return augmented

def apply_elastic_distortion(image, alpha=34, sigma=4):
    """
    Apply elastic distortion to the image.
    
    Parameters:
    - image: PIL Image object.
    - alpha: Scaling factor for distortion (higher = more distortion).
    - sigma: Standard deviation for the Gaussian filter.
    
    Returns:
    - Distorted PIL Image.
    """
    np_image = np.array(image).astype(np.float32)

    # Generate random displacement fields
    random_x = np.random.rand(*np_image.shape) * 2 - 1
    random_y = np.random.rand(*np_image.shape) * 2 - 1

    # Smooth the displacement fields
    from scipy.ndimage import gaussian_filter
    dx = gaussian_filter(random_x, sigma) * alpha
    dy = gaussian_filter(random_y, sigma) * alpha

    # Create meshgrid and apply distortions
    x, y = np.meshgrid(np.arange(np_image.shape[1]), np.arange(np_image.shape[0]))
    indices = np.array([y + dy, x + dx]).astype(np.float32)

    # Map coordinates to the image
    from scipy.ndimage import map_coordinates
    distorted_image = map_coordinates(np_image, indices, order=1, mode='reflect')

    return Image.fromarray(distorted_image.astype(np.uint8))

def add_gaussian_noise(image, mean=0, std=0.05):
    """
    Add Gaussian noise to the image.
    """
    np_image = np.array(image).astype(np.float32) / 255.0
    noise = np.random.normal(mean, std, np_image.shape)
    np_image += noise
    np_image = np.clip(np_image, 0.0, 1.0)
    return Image.fromarray((np_image * 255).astype(np.uint8))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MNIST PNG images to text format with optional data sampling and augmentation.")
    parser.add_argument("input_dir", type=str, help="Input directory containing MNIST data split into subsets (train, test) and labels (0-9).")
    parser.add_argument("output_dir", type=str, help="Output directory to save processed text files.")
    parser.add_argument("--percent", type=float, default=100.0, help="Percentage of data to process (0 < percent <= 100). Default is 100%.")
    parser.add_argument("--augment", action='store_true', help="Enable data augmentation to increase dataset size.")
    
    args = parser.parse_args()
    
    # Check the value of percent
    if args.percent <= 0 or args.percent > 100:
        print("The --percent argument must be in the range (0, 100].", file=sys.stderr)
        sys.exit(1)
    
    # Ensure percent is treated as a float
    percent = float(args.percent)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    preprocess(args.input_dir, args.output_dir, percent, args.augment)
