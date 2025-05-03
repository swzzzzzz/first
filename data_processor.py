import numpy as np
import cupy as cp
import logging

logger = logging.getLogger("DataProcessor")


def augment_batch(x_batch, t_batch):
    """
    Perform data augmentation on a batch of images and their labels.
    Ensures all operations stay on GPU using CuPy.

    Args:
        x_batch: Batch of images (N, C, H, W) as CuPy array
        t_batch: Batch of labels (N, classes) as CuPy array

    Returns:
        Augmented images and unchanged labels
    """
    # Ensure input is on GPU
    if not isinstance(x_batch, cp.ndarray):
        x_batch = cp.asarray(x_batch, dtype=cp.float32)

    # Get batch size
    batch_size = x_batch.shape[0]

    # Create a copy to avoid modifying the original data
    augmented_x = x_batch.copy()

    # Randomly apply augmentation to each image in the batch
    for i in range(batch_size):
        # Random horizontal flip (50% probability)
        if cp.random.rand() > 0.5:
            # Flip along width dimension
            augmented_x[i] = cp.flip(augmented_x[i], axis=2)

        # Random brightness adjustment
        brightness_factor = 0.9 + 0.2 * cp.random.rand()  # Range: [0.9, 1.1]
        augmented_x[i] = augmented_x[i] * brightness_factor

        # Ensure values stay in valid range
        augmented_x[i] = cp.clip(augmented_x[i], 0, 1)

    return augmented_x, t_batch


# Alias for backward compatibility
data_augmentation = augment_batch


# def data_augmentation(image, label):
#     image_hwc = image.transpose(1, 2, 0)
#     augmented_images = []
#     augmented_labels = []

#     augmented_images.append(image)
#     augmented_labels.append(label)

#     # 水平翻转
#     flipped = cp.flip(image_hwc, axis=1)
#     augmented_images.append(flipped.transpose(2, 0, 1))
#     augmented_labels.append(label)

#     # 垂直翻转
#     flipped_v = cp.flip(image_hwc, axis=0)
#     augmented_images.append(flipped_v.transpose(2, 0, 1))
#     augmented_labels.append(label)

#     brightness_factors = [0.8, 1.2]
#     for factor in brightness_factors:
#         brightness = cp.clip(image_hwc * factor, 0, 1)
#         augmented_images.append(brightness.transpose(2, 0, 1))
#         augmented_labels.append(label)

#     return augmented_images, augmented_labels


def apply_data_augmentation(x, t, augment_factor=3):
    logger.info(f"Applying data augmentation with factor {augment_factor}...")
    aug_images = []
    aug_labels = []

    for i in range(len(x)):
        img = x[i]
        lbl = t[i]
        new_images, new_labels = data_augmentation(img, lbl)
        aug_images.extend(new_images[:augment_factor])
        aug_labels.extend(new_labels[:augment_factor])

    x_augmented = cp.array(aug_images, dtype=cp.float32)
    t_augmented = cp.array(aug_labels, dtype=cp.float32)

    # 随机打乱
    x_combined = cp.concatenate([x, x_augmented], axis=0)
    t_combined = cp.concatenate([t, t_augmented], axis=0)

    indices = cp.random.permutation(len(x_combined))
    x_combined = x_combined[indices]
    t_combined = t_combined[indices]

    logger.info(
        f"Data augmentation complete. Augmented shape: {x_combined.shape}")
    return x_combined, t_combined


def normalize_data(x):
    # 使用mean 和 std 进行标准化
    mean = cp.mean(x, axis=(0, 2, 3), keepdims=True)
    std = cp.std(x, axis=(0, 2, 3), keepdims=True)
    std = cp.where(std < 1e-7, 1e-7, std)
    normalized_x = (x - mean) / std
    return normalized_x
