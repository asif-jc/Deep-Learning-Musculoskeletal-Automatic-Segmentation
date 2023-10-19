#Libraries
import numpy as np
import os
import pydicom
import SimpleITK as sitk
import pandas as pd
import trimesh
from pyntcloud import PyntCloud
from skimage.transform import resize
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib
import re
import cv2
from PIL import Image

# Helper Function
def flatten_3d_to_2d(array_3d):
    # Get the dimensions of the 3D array
    depth, height, width = array_3d.shape
    
    # Reshape the 3D array to a 2D array
    array_2d = np.reshape(array_3d, (depth, height * width))
    
    return array_2d

def transform_string(input_string):
    parts = input_string.split('_')
    
    if len(parts) < 2:
        return None
    
    prefix = parts[-2].upper()
    
    try:
        number_part = int(parts[0])
    except ValueError:
        return None
    
    new_string = f'{prefix}_{number_part:03d}'
    return new_string

# Helper Function
def flatten_2d_array(arr):
    flattened = []
    for row in arr:
        flattened.extend(row)
    return flattened
# Helper Function
# Read in entire scan of single patient
# folders = [f for f in os.listdir('MRI Scans - Tairawhiti') if os.path.isdir(os.path.join('MRI Scans - Tairawhiti', f))]
def ListFolders(directory):
    folder_names = []
    for root, dirs, files in os.walk(directory):
        for folder in dirs:
            folder_names.append(folder)
    return folder_names
# Helper Function
def read_dicom_files(directory):
    dicom_files = []
    files = os.listdir(directory)
    sorted_files = sorted(files)
    # print(sorted_files)
    for filename in sorted_files:
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath) and filename.endswith('.dcm'):
            try:
                dicom_file = pydicom.dcmread(filepath)
                dicom_files.append(dicom_file)
            except pydicom.errors.InvalidDicomError:
                print(f"Skipping file: {filename}. It is not a valid DICOM file.")
    return dicom_files
# Helper Function
def get_ram_usage(variable, variable_name):
    size_in_bytes = sys.getsizeof(variable)
    size_in_kb = size_in_bytes / 1024
    size_in_mb = size_in_kb / 1024
    size_in_gb = size_in_mb / 1024
    message = "Memory usage of %s: %d %s." % (variable_name, size_in_mb, 'MB')
    print(message)
#Helper Function
def convert_4d_to_3d(array_4d, axis):
    array_3d = np.squeeze(array_4d, axis=axis)
    return array_3d
def plot_3d_data(x, y, z):
    # Define the size of the figure (width, height) in inches
    fig = plt.figure(figsize=(4, 4))
    # Plot the 3D scatter plot
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='blue')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
def superimpose_images(image1, image2):
    image1 = image1 / np.max(image1)
    image2 = image2 / np.max(image2)
    alpha = 0.5
    superimposed_image = alpha * image1 + (1 - alpha) * image2
    return superimposed_image


def ReadIn_MRIScans_Masks(scans_path, folders):
    print('Patient Scan Data: ', folders)
    scan_pixel_data = []
    scan_coordinate_data = []
    single_scan_pixel_data = []
    scan_coordinate_data = []
    scan_orientation_data = []
    scan_pixelspacing_data = []

    single_paitent_scans_path =  scans_path + '/Raw DICOM MRI Scans/{}'.format(folders)
    dicom_files = read_dicom_files(single_paitent_scans_path)

    # Extracting pixel data
    for i in range (len(dicom_files)):
        normalized_data = (dicom_files[i].pixel_array)/(np.max(dicom_files[i].pixel_array))
        single_scan_pixel_data.append(normalized_data)
        print("Max pixel value in image stack is: ", np.max(normalized_data))
    scan_pixel_data.append(single_scan_pixel_data)

    training_scans = flatten_2d_array(scan_pixel_data)
    training_scans = np.array(training_scans)
    print("Max pixel value in image stack is: ", training_scans.max())
    
    # Coordinate Data
    single_paitent_scans_path =  scans_path + '/{}'.format(folders)
    for i in range (len(dicom_files)):
        scan_coordinate_data.append(dicom_files[i].ImagePositionPatient)
        scan_orientation_data.append(dicom_files[i].ImageOrientationPatient)
        scan_pixelspacing_data.append(dicom_files[i].PixelSpacing)
        
    coord_data = pd.DataFrame(scan_coordinate_data, columns=["x", "y", "z"])
    return training_scans


# Mapping coordinate data from groundtruth mask/label to mri training data
def MappingCoordinateData(filename_label, coord_data):
    # Load in mesh of label data
    mesh = trimesh.load_mesh(('C:/Users/GGPC/OneDrive/Desktop/Part 4 Project/Part4Project/SegmentationMasks/{}.ply').format(filename_label))

    # Convert the mesh vertices to a DataFrame
    vertices = pd.DataFrame(mesh.vertices, columns=["x", "y", "z"])

    # Pranav Ordering
    # coord_data = coord_data.sort_values('z', ascending = False)
    # coord_data = coord_data.reset_index(drop = True)
    vertices = vertices.sort_values('z', ascending = False)
    vertices = vertices.reset_index(drop = True)

    print('Height of Paitent in mm: ', np.abs(coord_data.iloc[-1][2] - coord_data.iloc[0][2]))
    print('Length of Paitent AOI (tibia) in mm: ', np.abs(vertices.iloc[-1][2] - vertices.iloc[0][2]))
    # vertices['z'] = vertices['z'].apply(lambda x: round(x, 1))
    coord_data['z'] = coord_data['z'].apply(lambda x: round(x, 1))

    # plot_3d_data(np.array(vertices['x']), np.array(vertices['y']), np.array(vertices['z']))

    if True:
        vertices.to_csv('ExactMaskCoordinateData.csv')
        coord_data.to_csv('ExactScanCoordinateData.csv')
        
    # vertices['z'] = np.round(vertices['z'] * 2) / 2
    # coord_data['z'] = np.round(coord_data['z'] * 2) / 2
    vertices['z'] = np.round(vertices['z'] * 2) / 2
    # coord_data['z'] = np.round(coord_data['z'] * 10) / 10
    # vertices['z'] = vertices['z'].apply(lambda x: round(x, 1))
    # coord_data['z'] = coord_data['z'].apply(lambda x: round(x, 1))
    if True:
        vertices.to_csv('RoundedMaskCoordinateData.csv')

    merged_df = pd.merge(coord_data, vertices, on='z')
    # condensed_df = merged_df.groupby('z').mean().reset_index()
    condensed_df = merged_df.groupby('z').median().reset_index()

    mapping_dict = dict(zip(condensed_df['z'], ['AOI']*len(condensed_df)))

    coord_data['SegmentationRegionSlice'] = coord_data['z'].map(mapping_dict).fillna('Outside of AOI')

    slices_aoi_start = (coord_data.loc[coord_data['SegmentationRegionSlice'] == 'AOI'].index)[0]
    slices_aoi_end = (coord_data.loc[coord_data['SegmentationRegionSlice'] == 'AOI'].index)[-1]
    slice_aoi_range = (slices_aoi_end - slices_aoi_start + 1)
    print('AOI Slice Start: ', slices_aoi_start)
    print('AOI Slice End: ', slices_aoi_end)
    print('AOI Slice Range: ', slice_aoi_range)

    # CSV Format 
    if True:
        coord_data.to_csv('tibia_mri_coord.csv')
        merged_df.to_csv('mergedcoordsystems.csv')
        condensed_df.to_csv('condensedmergedcoordsystems.csv')
    # print(coord_data)

    return slices_aoi_start, slices_aoi_end, slice_aoi_range, coord_data



def VoxelisationMask(filename_label, slice_aoi_range):
    # Load in mesh of label data
    mesh = trimesh.load_mesh(('C:/Users/GGPC/OneDrive/Desktop/Part 4 Project/Part4Project/SegmentationMasks/{}.ply').format(filename_label))

    # # Convert the mesh vertices to a DataFrame
    # vertices = pd.DataFrame(mesh.vertices, columns=["x", "y", "z"])
    # # Convert the mesh to a PyntCloud object
    # cloud = PyntCloud(vertices)
    vertices = pd.DataFrame(mesh.vertices, columns=["x", "y", "z"])
    faces = pd.DataFrame(mesh.faces, columns=['v1', 'v2', 'v3'])
    cloud = PyntCloud(points=vertices, mesh=faces)

    # Set the desired resolution
    desired_resolution = [slice_aoi_range, 512, 512]

    # Voxelize the mesh using the PyntCloud voxelization module
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=desired_resolution[0], n_y=desired_resolution[1], n_z=desired_resolution[2])
    voxel_grid = cloud.structures[voxelgrid_id].get_feature_vector().reshape(desired_resolution)

    # Transpose and swap axes to change the voxel grid orientation
    voxel_grid = np.transpose(voxel_grid, axes=(2, 0, 1))

    # Resize the voxel grid to match the desired dimensions
    voxel_grid = resize(voxel_grid, desired_resolution, anti_aliasing=False)

    voxel_grid = np.where(voxel_grid > 0, 1, 0)

    # print('Mask Slices Normalized to MRI Scans Shape (Purely AOI): ', voxel_grid.shape)

    return voxel_grid



def MaskCreation(basedir, filename_label):
    seg_masks_15A_tibia = sitk.ReadImage(('{}/Raw NIFITI Segmentation Masks (3D Slicer Output)/{}').format(basedir, filename_label))
    seg_masks_15A_tibia_data = sitk.GetArrayFromImage(seg_masks_15A_tibia)
    voxel_dimensions = seg_masks_15A_tibia.GetSpacing()
    voxel_dimensions = pd.DataFrame(data = voxel_dimensions)

    seg_masks_15A_tibia_data = seg_masks_15A_tibia_data[::-1, :, :]
    seg_masks_15A_tibia_data = np.where(seg_masks_15A_tibia_data != 0, 1, 0)

    seg_masks_15A_tibia_data = np.array(seg_masks_15A_tibia_data)
    indices = np.transpose(np.nonzero(seg_masks_15A_tibia_data != 0))
    if indices.size > 0:
        first_non_zero_index_2d = tuple(indices[0])
    else:
        first_non_zero_index_2d = None

    if indices.size > 0:
        last_non_zero_index_2d = tuple(indices[-1])
    else:
        last_non_zero_index_2d = None

    print("AOI Slice Start: ", first_non_zero_index_2d[0])
    print("AOI Slice End: ", last_non_zero_index_2d[0])
    print("AOI Slice Range: ", (last_non_zero_index_2d[0] - first_non_zero_index_2d[0] + 1))

    return seg_masks_15A_tibia_data, first_non_zero_index_2d[0], last_non_zero_index_2d[0]

def VisualValidationMSK(basedir, colab_fname, slice_idx_bin, slice_idx_multi, multiclass_dir, mask_index):
    nii_img_scan = nib.load(('{}/nnUNet Data/scans/msk_00{}.nii.gz').format(basedir, int(((colab_fname[0]).split('_'))[1])))
    nii_img_mask = nib.load(('{}/nnUNet Data/masks/{}/{}.nii.gz').format(basedir, ((colab_fname[0]).split('_'))[0],colab_fname[0]))

    mask_data = nii_img_mask.get_fdata()
    scan_data = nii_img_scan.get_fdata()
    image1 = scan_data[int(slice_idx_bin), :, :, 0]
    image2 = mask_data[int(slice_idx_bin), :, :, 0]
    superimposed_image = superimpose_images(image1, image2)
    plt.imshow(superimposed_image, cmap='gray')
    plt.title(('Validating Scan & Mask (Binary) on Slice {} of {}').format(slice_idx_bin, colab_fname[0]))
    plt.axis('off')
    plt.show()

    if (multiclass_dir != False):
        nii_img_mask_multiclass = nib.load(('{}/msk_00{}.nii.gz').format(multiclass_dir, mask_index))
        multiclass_mask_data = nii_img_mask_multiclass.get_fdata()
        image1 = scan_data[slice_idx_multi, :, :, 0]
        image2 = multiclass_mask_data[slice_idx_multi, :, :, 0]
        superimposed_image = superimpose_images(image1, image2)
        plt.imshow(superimposed_image, cmap='gray')
        plt.title(('Validating Scan & Mask (Multiclass) on Slice {}/{} of msk_00{}').format(slice_idx_multi, multiclass_mask_data.shape[0], mask_index))
        plt.axis('off')
        plt.show()


def preprocessing(scans_path, filename_labels, folders, total_slices_raw_data, DataOnlyAOI, Cropping):
    train_mask_tibia_labels, training_scans, start_slices_aoi, end_slices_aoi, slice_aoi_ranges  = [], [], [], [], []
    filename_labels = [filename_labels]
    print('\n')
    print('Patient Scan Data Folders Included in Run: ', folders)

    for index, filename_label in enumerate(filename_labels):
        print('\n')
        print('Segmentation Mask: ',('{}'.format(filename_label)))

        training_scan = ReadIn_MRIScans_Masks(scans_path, folders[index])
        seg_masks_15A_tibia_data, slices_aoi_start, slices_aoi_end = MaskCreation(scans_path, filename_label)
        median_aoi_index = int(np.abs(slices_aoi_end - slices_aoi_start) / 2) + slices_aoi_start

        if(DataOnlyAOI == True):
            if (index == 0):
                train_mask_tibia_labels = seg_masks_15A_tibia_data[(slices_aoi_start):(slices_aoi_end+1)]
                training_scans = training_scan[(slices_aoi_start):(slices_aoi_end+1)]
            else:
                train_mask_tibia_labels = np.concatenate((train_mask_tibia_labels, seg_masks_15A_tibia_data[(slices_aoi_start):(slices_aoi_end+1)]), axis=0)
                training_scans = np.concatenate((training_scans, training_scan[(slices_aoi_start):(slices_aoi_end+1)]), axis=0)

        if (DataOnlyAOI == False):
            train_mask_tibia_labels.append(seg_masks_15A_tibia_data)
            training_scans.append(training_scan)

        start_slices_aoi.append(slices_aoi_start)
        end_slices_aoi.append(slices_aoi_end)

        print('\n')

    train_mask_tibia_labels = np.array(train_mask_tibia_labels)
    training_scans = np.array(training_scans)

    # Normalization and Binarization
    training_scans = training_scans.astype('float32')
    training_scans /= 255.  # scale scans to [0, 1]
    train_mask_tibia_labels = np.where(train_mask_tibia_labels != 0, 1, 0)
    print("Scans Normalized! [0-1]")
    print("Max pixel value in image stack is: ", training_scans.max())
    print("Masks Binarised! [0,1]")
    print("Labels in the mask are : ", np.unique(train_mask_tibia_labels))
    print("\n")

    if (DataOnlyAOI == False):
        training_scans = np.reshape(training_scans, (len(folders) * training_scans.shape[1], 512, 512))
        train_mask_tibia_labels = np.reshape(train_mask_tibia_labels, (len(filename_labels) * train_mask_tibia_labels.shape[1], 512, 512))
        training_scans = np.expand_dims(training_scans, axis=-1)
        train_mask_tibia_labels = np.expand_dims(train_mask_tibia_labels, axis=-1)

    if (DataOnlyAOI == True):
        training_scans = np.expand_dims(training_scans, axis=-1)
        train_mask_tibia_labels = np.expand_dims(train_mask_tibia_labels, axis=-1)


    if (Cropping == True):
    # Resize 'training_scans'
        training_scans_resized = np.empty((training_scans.shape[0], 256, 256, 1), dtype=np.float32)
        for i in range(training_scans.shape[0]):
            input_image = training_scans[i, :, :, 0]  # Extract the 2D image from the 4D array
            pil_image = Image.fromarray(input_image)  # Convert to Pillow Image
            resized_image = pil_image.resize((256, 256), Image.BILINEAR)  # Resize using Pillow
            training_scans_resized[i, :, :, 0] = np.array(resized_image)  # Store the resized image in the output array


        # Resize 'train_mask_tibia_labels'
        train_mask_tibia_labels_resized = np.empty((train_mask_tibia_labels.shape[0], 256, 256, 1), dtype=np.uint8)
        for i in range(train_mask_tibia_labels.shape[0]):
            input_image = train_mask_tibia_labels[i, :, :, 0]  # Extract the 2D image from the 4D array
            pil_image = Image.fromarray(input_image)  # Convert to Pillow Image
            resized_image = pil_image.resize((256, 256), Image.BILINEAR)  # Resize using Pillow
            train_mask_tibia_labels_resized[i, :, :, 0] = np.array(resized_image)  # Store the resized image in the output array

        training_scans = training_scans_resized
        train_mask_tibia_labels = train_mask_tibia_labels_resized

    print('Training Scans Input Shape (Full 3D Stack): ', training_scans.shape)
    print('Training Masks Input Shape (Full 3D Stack): ', train_mask_tibia_labels.shape)

    return training_scans, train_mask_tibia_labels, median_aoi_index