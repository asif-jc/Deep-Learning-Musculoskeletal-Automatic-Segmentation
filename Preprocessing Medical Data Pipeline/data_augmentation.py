import numpy as np
import cv2
from random import randint
from tqdm import tqdm


def flatten_array(arr):
    return arr.reshape(-1, *arr.shape[2:])

def ImageDataAugmentation(img, mask, num_augmentations):
    augmented_images = []
    augmented_masks = []

    for i in range(num_augmentations):
        
        # Apply a random rotation to the images
        angle = randint(-15, 15)
        M = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)
        rotated_img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        rotated_mask = cv2.warpAffine(mask, M, (mask.shape[1], mask.shape[0]))

        # Apply a random horizontal flip to the images
        if randint(0, 1):
            flipped_img = cv2.flip(rotated_img, 1)
            flipped_mask = cv2.flip(rotated_mask, 1)
        else:
            flipped_img = rotated_img
            flipped_mask = rotated_mask

        # Append the augmented images and masks to the lists
        augmented_images.append(flipped_img)
        augmented_masks.append(flipped_mask)

    # Convert the lists of augmented images and masks to NumPy arrays
    augmented_images = np.array(augmented_images)
    augmented_masks = np.array(augmented_masks)

    return augmented_images, augmented_masks

def DataAugmentation (imgs_train, imgs_mask_train, num_augmentations):
    augmented_images_train = []
    augmented_masks_train = []

    for i in tqdm(range(len(imgs_train))):
        augmented_images, augmented_masks = ImageDataAugmentation(imgs_train[i,:,:,0], imgs_mask_train[i,:,:,0], num_augmentations)
        augmented_images_train.append(augmented_images)
        augmented_masks_train.append(augmented_masks)

    augmented_images_train = np.array(augmented_images_train)
    augmented_masks_train = np.array(augmented_masks_train)

    augmented_images_train = flatten_array(augmented_images_train)
    augmented_masks_train = flatten_array(augmented_masks_train)

    return augmented_images_train, augmented_masks_train