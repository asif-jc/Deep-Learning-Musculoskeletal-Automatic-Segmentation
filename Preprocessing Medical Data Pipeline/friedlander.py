import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from MSKMulticlass import CreateMasks4MulticlassMSK

def superimpose_images(image1, image2):
    image1 = image1 / np.max(image1)
    image2 = image2 / np.max(image2)
    alpha = 0.5
    superimposed_image = alpha * image1 + (1 - alpha) * image2
    return superimposed_image

def uniform_resizing(images, new_size):
    """
    Applies uniform resizing to an array of images. ### bicubic_interpolation

    Parameters:
        images (numpy.ndarray): Input array of images with shape (x, 512, 512, 1).
        new_size (tuple or int): The desired size of the resized images. If it's an int,
                                 the new size will be (new_size, new_size).

    Returns:
        numpy.ndarray: Array of resized images with shape (x, new_height, new_width, 1).
    """
    x, height, width, _ = images.shape

    if isinstance(new_size, int):
        new_height, new_width = new_size, new_size
    elif isinstance(new_size, tuple) and len(new_size) == 2:
        new_height, new_width = new_size
    else:
        raise ValueError("Invalid new_size argument. It should be an int or a tuple of two ints.")

    resized_images = np.zeros((x, new_height, new_width, 1))

    for i in range(x):
        resized_images[i, ..., 0] = resize(images[i, ..., 0], (new_height, new_width), mode='reflect', order=1)
        # resized_images[i, ..., 0] = resize(images[i, ..., 0], (new_height, new_width), mode='constant', preserve_range=True)

    return resized_images







# Pranav Change!

runVisualisationONLY = True

# Base Directory (Make a new folder!)
base_dir = 'D:/MRI - Friedlander'

# The mask index is the msk_00X number
# mask_index = [1] # Does one at a time
mask_index = [2,3,4,5] # Does multiple

# Make the following folders from your base directory
# 1)    First make a folder called nnUNet Data 
# 2)    From nnUNet folder make: scans, masks, multiclass_masks
# 3)    For unprocessed_scans folder load tbe scans as nii.gz using the msk_00X naming convention (ALWAYS back up scans outside of this file it )
# 4)    For masks folder look on discord for a reference pic
# 5)    For scans folder leave empty for processed out files
individual_mask_directory = ('{}/nnUNet Data/masks').format(base_dir)
multiclass_mask_output_dir = ('{}/nnUNet Data/multiclass_masks').format(base_dir)
scan_dir = ('{}/nnUNet Data/scans').format(base_dir) #output scans directory not unprocessed scans
input_scan_dir = ('{}/nnUNet Data/unprocessed_scans').format(base_dir)

# DONT EDIT!
TIBIA_encoding = 1
FEMUR_encoding = 2
FIBULA_encoding = 3
PELVIS_encoding = 4
AOIThresholding = True
FriedLanderDataset = True

if (runVisualisationONLY == False):
    print('Multi-Class Segmentation Task Data Preparation!')
    for i in range (len(mask_index)):
        CreateMasks4MulticlassMSK(input_scan_dir, scan_dir, individual_mask_directory, mask_index[i], TIBIA_encoding, FEMUR_encoding, FIBULA_encoding, PELVIS_encoding, multiclass_mask_output_dir, AOIThresholding, FriedLanderDataset)


# Visualisations

# Pranav Change!
slice = 650
msk_index = 3

img = nib.load(('{}/msk_00{}.nii.gz').format(scan_dir, msk_index))
img_data = img.get_fdata()
print('Training Scan Shape: ', img_data.shape)

mask = nib.load(('{}/msk_00{}.nii.gz').format(multiclass_mask_output_dir, msk_index))
mask_data = mask.get_fdata()
print('Training Mask Shape: ', mask_data.shape)

superimposed_data = superimpose_images(img_data[slice, :, :, 0], mask_data[slice, :, :, 0])

plt.imshow(img_data[slice, :, :, 0], cmap='gray')
plt.axis('off')
plt.show()

plt.imshow(mask_data[slice, :, :, 0], cmap='gray')
plt.axis('off')
plt.show()

plt.imshow(superimposed_data, cmap='gray')
plt.axis('off')
plt.show()