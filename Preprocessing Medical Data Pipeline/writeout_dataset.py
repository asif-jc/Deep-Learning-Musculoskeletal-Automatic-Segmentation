# Exporting Training Scan and Mask Data
import os 
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import re
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm

def WriteOutTextFile(data_array, save_dir, starting_slice):
    num_files = data_array.shape[0]
    
    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    for i in range(num_files):
        file_name = f"data_{i + starting_slice}.txt"
        file_path = save_dir + '/' + file_name
        np.savetxt(file_path, data_array[i], delimiter=',', fmt='%.4f')
        print(f"Saved {file_path}")


def WriteOutImagePNGFiles(data_array, save_dir, starting_slice):
    num_files = data_array.shape[0]

    # Create the save directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i in range(num_files):
        file_name = f"image_{i + starting_slice}.png"
        file_path = save_dir + '/' + file_name
            
        # Scale the pixel values to the range [0, 255] if necessary
        image_data = ((data_array[i] * 255).astype(np.uint8)).squeeze()
        # Create an Image object and save as PNG
        image = Image.fromarray(image_data)

        if True:
            image.save(file_path)
                
        print(f"Saved {file_path}")


def ReadInDatasets(folder_path):
    files = os.listdir(folder_path)
    num_files = len(files)
    result_array = np.empty((num_files, 512, 512))

    for i, file in enumerate(files):
        file_path = os.path.join(folder_path, file)
        with open(file_path, 'r') as f:
            # Read the contents of the text file
            content = f.read()

            # Convert the content into a 2D NumPy array of floats
            array_2d = np.array([list(map(float, line.split(','))) for line in content.splitlines()])

            # Store the 2D array in the result array
            result_array[i] = array_2d

    return result_array

def ReadInDatasetNPY(folder_path):
    # Get a list of all files in the folder
    file_list = os.listdir(folder_path)

    # Filter out only the .npy files
    npy_files = [file for file in file_list if file.endswith('.npy')]

    # Sort the .npy files based on the numeric order in their filenames
    npy_files.sort(key=lambda x: int(x.split('_')[0]))

    # Initialize an empty list to store arrays from each .npy file
    arrays_list = []

    # Read in each .npy file and append it to the list
    for file_name in npy_files:
        file_path = os.path.join(folder_path, file_name)
        array = np.load(file_path)
        arrays_list.append(array)

    # Concatenate the arrays into a single array
    concatenated_array = np.concatenate(arrays_list, axis=0)

    return concatenated_array

def Export2CompressedNifiti(imgs_train, scans_path, colab_fname, imgs_mask_train, orientation):
    nii_img_train = nib.Nifti1Image(imgs_train, affine=np.eye(4))
    scan_idx_fnamecolab = int((colab_fname[0].split('_'))[1])
    scan_idx_fnamecolab = '{:03d}'.format(scan_idx_fnamecolab)
    output_file_path = ('{}/nnUNet Data/unprocessed_scans/msk_{}.nii.gz').format(scans_path, scan_idx_fnamecolab)
    nib.save(nii_img_train, output_file_path)

    nii_img_mask = nib.Nifti1Image(imgs_mask_train, affine=np.eye(4))
    output_file_path = ('{}/nnUNet Data/masks/{}/{}_{}.nii.gz').format(scans_path, ((colab_fname[0]).split('_'))[0],colab_fname[0], orientation)
    nib.save(nii_img_mask, output_file_path)   

    print("Conversion to NIfTI complete with shape: ", imgs_mask_train.shape)
    print('\n')
    print('Successfully Exported Preprocessed Data!')
