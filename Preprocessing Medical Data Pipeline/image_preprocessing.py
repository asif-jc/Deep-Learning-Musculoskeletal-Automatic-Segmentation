
import numpy as np
from skimage.transform import resize

def image_cropping(images, top, bottom, left, right):
    x, height, width, _ = images.shape

    if top + bottom >= height or left + right >= width:
        raise ValueError("Total cropping size cannot exceed the original image size.")

    cropped_images = np.zeros((x, height - top - bottom, width - left - right, 1))

    for i in range(x):
        cropped_images[i] = images[i, top:height - bottom, left:width - right]

    return cropped_images

def uniform_cropping(images, crop_size):
    """
    Applies uniform cropping to an array of images.

    Parameters:
        images (numpy.ndarray): Input array of images with shape (x, 512, 512, 1).
        crop_size (int): The desired size of the square crop.

    Returns:
        numpy.ndarray: Array of cropped images with shape (x, crop_size, crop_size, 1).
    """
    x, height, width, _ = images.shape

    if crop_size > height or crop_size > width:
        raise ValueError("Crop size cannot be larger than the original image size.")

    # Calculate the top-left corner for cropping to keep it uniform
    top_left_y = (height - crop_size) // 2
    top_left_x = (width - crop_size) // 2

    cropped_images = np.zeros((x, crop_size, crop_size, 1))

    for i in range(x):
        cropped_images[i] = images[i, top_left_y:top_left_y + crop_size, top_left_x:top_left_x + crop_size]

    return cropped_images

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