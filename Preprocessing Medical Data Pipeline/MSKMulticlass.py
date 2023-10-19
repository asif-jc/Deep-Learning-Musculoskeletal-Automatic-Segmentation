import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
from image_preprocessing import uniform_resizing

def CreateMasks4MulticlassMSK(input_scan_dir, scan_dir, directory, mask_index, TIBIA_encoding, FEMUR_encoding, FIBULA_encoding, PELVIS_encoding, output_dir, AOIThresholding, FriedLanderDataset):
    aoi_fnames = os.listdir(directory)
    resizing_size = 256
    print('Regions of Interest for Segmentation: ', aoi_fnames)

    fnames = []
    for aoi_fname in aoi_fnames:
        directory_aoi = (directory + ('/{}').format(aoi_fname))
        aoi_fname = os.listdir(directory_aoi)
        aoi_fname = sorted(aoi_fname)
        fnames.append(aoi_fname)

    n_segmentation_classes = len(fnames)
    print('Number of Segmentation Classes: ', n_segmentation_classes)



    suffix_index_dict = {}
    for idx, file_list in enumerate(fnames):
        for filename in file_list:
            # Extract the suffix index from the filename
            parts = filename.split('_')
            if len(parts) > 1:
                suffix_index = parts[1].split('.')[0]  # Extracting the index before '.'
                suffix_index = int(suffix_index)  # Convert index to integer
                # Add the filename to the corresponding list in the dictionary
                if suffix_index in suffix_index_dict:
                    suffix_index_dict[suffix_index].append(filename)
                else:
                    suffix_index_dict[suffix_index] = [filename]

    # for index, filenames in suffix_index_dict.items():
    #     print(f"Segmentation Mask {index}: {filenames}")

    combined_mask = None
    for fname_mask in suffix_index_dict[mask_index]:
        print(fname_mask)
        mask = nib.load(('{}/{}/{}').format(directory, (fname_mask.split('_'))[0], fname_mask))
        mask_data = mask.get_fdata().astype(np.uint8)
        aoi = (fname_mask.split('_'))[0]

        if (FriedLanderDataset == True):
            mask_data = np.expand_dims(mask_data, axis = -1)
            mask_data = mask_data.transpose(1, 0, 2, 3)
            print(('Unprocessed {} Mask Shape: {}').format(aoi, mask_data.shape))
            mask_data = uniform_resizing(mask_data, resizing_size)


        if (aoi == 'TIBIA'):
            mask_data = np.where(mask_data != 0, TIBIA_encoding, 0)
        if (aoi == 'FEMUR'):
            mask_data = np.where(mask_data != 0, FEMUR_encoding, 0)
        if (aoi == 'FIBULA'):
            mask_data = np.where(mask_data != 0, FIBULA_encoding, 0)
        if (aoi == 'PELVIS'):
            mask_data = np.where(mask_data != 0, PELVIS_encoding, 0)    
        if combined_mask is None:
            combined_mask = mask_data
        else:
            combined_mask += mask_data  # Combine the pixel arrays by adding them element-wise
            combined_mask = np.array(combined_mask)
            combined_mask[combined_mask > n_segmentation_classes] = 0

    if (AOIThresholding == True):
        threshold = 0

        combined_mask = np.array(combined_mask)
        indices = np.transpose(np.nonzero(combined_mask != 0))
        if indices.size > 0:
            first_non_zero_index_2d = tuple(indices[0])
        else:
            first_non_zero_index_2d = None

        if indices.size > 0:
            last_non_zero_index_2d = tuple(indices[-1])
        else:
            last_non_zero_index_2d = None


        print(first_non_zero_index_2d)
        print(last_non_zero_index_2d)
        first_slice_aoi = (first_non_zero_index_2d[0]) - int(threshold)
        last_slice_aoi = (last_non_zero_index_2d[0]) + int(threshold)
        print("AOI Slice Start (Final with Thresholding): ", first_slice_aoi)
        print("AOI Slice End (Final with Thresholding): ", last_slice_aoi)

        mask_index = '{:03d}'.format(mask_index)
        combined_mask = combined_mask[first_slice_aoi:last_slice_aoi, :, :, :]
        scan = nib.load(('{}/msk_{}.nii.gz').format(input_scan_dir, mask_index))
        scan_data = scan.get_fdata()
        # scan_og = nib.Nifti1Image(scan_data, scan.affine)
        # nib.save(scan_og, ('{}/msk_{}_og.nii.gz').format(output_dir, mask_index))

        if (FriedLanderDataset == True):
            scan_data = np.expand_dims(scan_data, axis = -1)
            scan_data = scan_data.transpose(1, 0, 2, 3) 
            print('Unprocessed Scan Shape: ', scan_data.shape)
            scan_data = uniform_resizing(scan_data, resizing_size)
            scan_data = scan_data.astype('float32')
            scan_data /= np.max(scan_data)  # scale scans to [0, 1]
            print('Max Pixel Value in Scan: ', np.max(scan_data))

        scan_data = scan_data[first_slice_aoi:last_slice_aoi, :, :, :]
        print('Final Training Scans Input Shape: ', scan_data.shape)
        print('Final Training Masks Input Shape: ', combined_mask.shape)

    # np.savetxt('output.txt', scan_data[100,:,:,0], fmt="%d", delimiter=",")
    combined_mask = combined_mask.astype(int)
    combined_img = nib.Nifti1Image(combined_mask, mask.affine)
    print('Multi-class Labels: ', np.unique(combined_mask))
    nib.save(combined_img, ('{}/msk_{}.nii.gz').format(output_dir, mask_index))

    combined_img = nib.Nifti1Image(scan_data, scan.affine)
    nib.save(combined_img, ('{}/msk_{}.nii.gz').format(scan_dir, mask_index))

    print(('MSK Multiclass Mask Made Using {}, Saved To {}, and Region of Interest Slice Thresholding = {} !').format(aoi_fnames, ('{}/msk_{}.nii.gz').format(output_dir, mask_index), AOIThresholding))
    print('\n')