import subprocess

required_libraries = [
    "numpy", "os", "pydicom", "SimpleITK", "pandas", "trimesh",
    "pyntcloud", "scikit-image", "matplotlib", "tqdm", "nibabel",
    "re", "scipy", "opencv-python", 'segmentation_models', 'random',
    'trimesh'
]

for lib in required_libraries:
    try:
        __import__(lib)
    except ImportError:
        print(f"{lib} is not installed. Installing...")
        subprocess.check_call(["pip", "install", lib])
        print(f"{lib} has been installed.")


from preprocessing import preprocessing, VisualValidationMSK, transform_string
from image_preprocessing import image_cropping, uniform_cropping, uniform_resizing
from writeout_dataset import Export2CompressedNifiti
from MSKMulticlass import CreateMasks4MulticlassMSK
from createDirectories import createDirectoriesfunc
from preparingTestInstance import preprocessTestScans
from data_augmentation import DataAugmentation
import os
# import tensorflow
# from tensorflow import keras
# from keras.models import load_model
# from keras.utils import to_categorical
import nibabel as nib
import numpy as np
# import segmentation_models as sm
# from keras.metrics import MeanIoU
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
import trimesh
import SimpleITK as sitk
import nibabel as nib
import numpy as np

createDirectoriesfunc()

def Preprocessing(scans_path, scan_data_folders, Cropping):
    print('-'*30)
    print('Loading and preprocessing training data...')
    print('-'*30)

    # /research\resmed202100086-tws ----> Address for raw data
    # /research\resabi202200010-Friedlander
    total_slices_raw_data = 0
    DataOnlyAOI = False
    ExportDatasets = True
    multiclass_mask_output_dir = None

    # Only set multiclassSegmentation to True once all the AOI masks are preprocessed or its onto the final AOI
    if (Cropping == "256x256"):
        Cropping = True
    else:
        Cropping = False

    scan_data_folders = [scan_data_folders]

    # Creates the preprocessed scan and mask data and exports to  Data folder
    # Formats binary masks and applies preprocessing steps 
    segmasks = []
    paitent_id = (scan_data_folders[0].split('_'))[-1]
    files = os.listdir(('{}/Raw NIFITI Segmentation Masks (3D Slicer Output)').format(scans_path))

    for file in files:
        file_id = file.split('_')[-1]
        file_id = file_id.split('.')[0]
        if (file_id == paitent_id):
            segmasks.append(file)
    print(segmasks)
    for segmask in segmasks:
        print(segmask)
        imgs_train, imgs_mask_train, median_aoi_index = preprocessing(scans_path, segmask, scan_data_folders, total_slices_raw_data, DataOnlyAOI, Cropping)
        orientation = (segmask.split('_'))[1]
        colab_fname = [transform_string(segmask)]
        Export2CompressedNifiti(imgs_train, scans_path, colab_fname, imgs_mask_train, orientation)

    # User Input
    # Multi-class mask creation
    individual_mask_directory = ('{}/nnUNet Data/masks').format(scans_path)
    multiclass_mask_output_dir = ('{}/nnUNet Data/multiclass_masks').format(scans_path)
    scan_dir = ('{}/nnUNet Data/scans').format(scans_path)
    input_scan_dir = ('{}/nnUNet Data/unprocessed_scans').format(scans_path)

    TIBIA_encoding = 1
    FEMUR_encoding = 2
    FIBULA_encoding = 3
    PELVIS_encoding = 4
    mask_index = int((scan_data_folders[0].split('_'))[0])
    AOIThresholding = True
    FriedLanderDataset = False

    print('Multi-Class Segmentation Task Data Preparation!')
    CreateMasks4MulticlassMSK(input_scan_dir, scan_dir, individual_mask_directory, mask_index, TIBIA_encoding, FEMUR_encoding, FIBULA_encoding, PELVIS_encoding, multiclass_mask_output_dir, AOIThresholding, FriedLanderDataset)


    print('-'*30)
    print('Completed Preprocessing Stage!')
    print('-'*30)

def preprocessTestScansMain(cutoffSlice, seg_scan_dir, folders):
    preprocessTestScans(cutoffSlice, seg_scan_dir, [folders])

def VisualiseSlice(basedir, colab_fname, slice_idx_bin):
    import nibabel as nib
    import numpy as np
    import matplotlib.pyplot as plt
    def superimpose_images(image1, image2):
        image1 = image1 / np.max(image1)
        image2 = image2 / np.max(image2)
        alpha = 0.5
        superimposed_image = alpha * image1 + (1 - alpha) * image2
        return superimposed_image
    
    if ((colab_fname.split('_'))[0] != 'msk'):
        orientation = (colab_fname.split('_'))[1]
        colab_fname = transform_string(colab_fname)
        mask_index = '{:03d}'.format(int(((colab_fname).split('_'))[1]))
        nii_img_scan = nib.load(('{}/nnUNet Data/scans/msk_{}.nii.gz').format(basedir, mask_index))
        nii_img_mask = nib.load(('{}/nnUNet Data/masks/{}/{}_{}.nii.gz').format(basedir, ((colab_fname).split('_'))[0],colab_fname, orientation))

    if ((colab_fname.split('_'))[0] == 'msk'):
        nii_img_scan = nib.load(('{}/nnUNet Data/scans/{}.nii.gz').format(basedir, colab_fname))
        nii_img_mask = nib.load(('{}/nnUNet Data/multiclass_masks/{}.nii.gz').format(basedir, colab_fname))
        orientation = '_'


    mask_data = nii_img_mask.get_fdata()
    scan_data = nii_img_scan.get_fdata()
    image1 = scan_data[int(slice_idx_bin), :, :, 0]
    image2 = mask_data[int(slice_idx_bin), :, :, 0]
    superimposed_image = superimpose_images(image1, image2)

    plt.imshow(superimposed_image, cmap='gray')
    plt.title(('Validating Scan & Mask on Slice {} of {}_{}').format(slice_idx_bin, colab_fname, orientation))
    plt.axis('off')
    plt.show()

    plt.imshow(image2, cmap='gray')
    plt.title(('Validating Mask on Slice {} of {}_{}').format(slice_idx_bin, colab_fname, orientation))
    plt.axis('off')
    plt.show()

    plt.imshow(image1, cmap='gray')
    plt.title(('Validating Scan on Slice {} of {}_{}').format(slice_idx_bin, colab_fname, orientation))
    plt.axis('off')
    plt.show()

    print('-'*30)
    print('Visulisations Generated!')
    print('-'*30)

def DataAug(basedir, aug_fname, num_augs):
    imgs_train = nib.load(('{}/nnUNet Data/scans/{}.nii.gz').format(basedir, aug_fname))
    imgs_mask_train = nib.load(('{}/nnUNet Data/multiclass_masks/{}.nii.gz').format(basedir, aug_fname))
    imgs_train = imgs_train.get_fdata()
    imgs_mask_train = imgs_mask_train.get_fdata()
    num_train = len(imgs_train)
    num_augs = int(num_augs)
    augmented_images_train, augmented_masks_train = DataAugmentation(imgs_train, imgs_mask_train, num_augs)

    augmented_images_train_sorted, augmented_masks_train_sorted = [], []
    for i in range (int(num_augs)):
        augmented_images_train_temp = augmented_images_train[i::num_augs]
        augmented_masks_train_temp = augmented_masks_train[i::num_augs]
        augmented_images_train_sorted.append(augmented_images_train_temp)  
        augmented_masks_train_sorted.append(augmented_masks_train_temp)  
    augmented_images_train_sorted = np.array(augmented_images_train_sorted)
    augmented_masks_train_sorted = np.array(augmented_masks_train_sorted)

    print('Augmented Training Scan Shape: ', augmented_images_train_sorted.shape)
    print('Augmented Training Mask Shape: ',augmented_masks_train_sorted.shape)


    for i in range (len(augmented_images_train_sorted)):
        temp = np.expand_dims(augmented_images_train_sorted[i], axis=-1)
        temp = temp.astype('float32')
        np.savetxt('D:\MRI - Tairawhiti (User POV)/train.txt', temp[100,:,:,0], fmt="%d", delimiter=",")
        # temp /= 255.  # scale scans to [0, 1]
        nii_img_train = nib.Nifti1Image(temp, affine=np.eye(4))
        output_file_path = ('{}/nnUNet Data/scans/{}_aug{}.nii.gz').format(basedir, aug_fname, i)
        nib.save(nii_img_train, output_file_path)

        temp = np.expand_dims(augmented_masks_train_sorted[i], axis=-1)
        temp = temp.astype(int)
        np.savetxt('D:\MRI - Tairawhiti (User POV)/train_mask.txt', temp[100,:,:,0], fmt="%d", delimiter=",")
        nii_img_mask = nib.Nifti1Image(temp, affine=np.eye(4))
        output_file_path = ('{}/nnUNet Data/multiclass_masks/{}_aug{}.nii.gz').format(basedir, aug_fname, i)
        nib.save(nii_img_mask, output_file_path)  

    print('-'*30)
    print('Data Augmentation Completed & Exported!')
    print('-'*30) 


def AutoSegModel(subjectfname, base_dir):
    model_dir = ('{}/Pre-Trained Models (Google Colab)/res34_backbone_20_epochs_dicefocal_256_4P_12batch_maxF1_pelvis_aug_best.hdf5').format(base_dir)
    model = load_model(model_dir, compile=False)

    n_classes = 5
    BACKBONE = 'resnet34'
    model_used = 'U-Net(resnet34)'

    img_dir = ('{}/nnUNet Data/scans/{}.nii.gz').format(base_dir, subjectfname)
    img = nib.load(img_dir)
    img_data = img.get_fdata()
    X_test = np.repeat(img_data, 3, axis=3)

    print(('Fined-Tuned Model: {}').format((model_dir.split('/'))[-1]))
    print('Prediction Scan Stack: ', subjectfname)
    print('Number of Segmentation Classes: ', n_classes)
    print('\n')

    print("Test Images Shape: ", X_test.shape)
    print('\n')

    preprocess_input = sm.get_preprocessing(BACKBONE)
    X_test_processed = preprocess_input(X_test)

    # Prediction
    y_pred=model.predict(X_test_processed)
    y_pred_argmax=np.argmax(y_pred, axis=3)
    y_pred_argmax = np.expand_dims(y_pred_argmax, axis = -1)
    print('Pred Mask Shape: ', y_pred_argmax.shape)
    print("Pred Mask Labels: ", np.unique(y_pred_argmax))
    print('\n')

    combined_mask = y_pred_argmax.astype(np.int32)
    combined_img = nib.Nifti1Image(combined_mask, affine=np.eye(4), dtype=np.int32)
    nib.save(combined_img, ("{}/{}_pred.nii.gz").format(model_dir, subjectfname), dtype = np.uint8)
    print('Exported Prediction Segmentation: ', (("{}/{}_pred.nii.gz").format(model_dir, subjectfname)))
    print('\n')

    def Export3DStructure(segmentation_data, predfname, class_msk, model_used):
        # Generate a surface mesh using marching cubes
        vertices, faces, normals, _ = marching_cubes(segmentation_data, level=0)

        # Create a Trimesh object
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

        # Save the mesh as a PLY file
        ply_path = ('{}/{}_{}_{}.ply').format(model_dir, predfname, class_msk, model_used)
        mesh.export(ply_path)

    segmentation_data_all = y_pred_argmax
    segmentation_data_all = segmentation_data_all[:,:,:,0]
    tibia_seg_data = np.where(segmentation_data_all != 1, 0, 1)
    femur_seg_data = np.where(segmentation_data_all != 2, 0, 2)
    fibula_seg_data = np.where(segmentation_data_all != 3, 0, 3)
    pelvis_seg_data = np.where(segmentation_data_all != 4, 0, 4)

    segmentation_data = [segmentation_data_all, tibia_seg_data, femur_seg_data, fibula_seg_data, pelvis_seg_data]
    class_msk = ['ALL', 'TIBIA', 'FEMUR', 'FIBULA', 'PELVIS']

    for i in range (len(segmentation_data)):
        Export3DStructure(segmentation_data[i], subjectfname, class_msk[i], model_used)

