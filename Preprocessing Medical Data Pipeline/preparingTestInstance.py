# from preprocessing import  ReadIn_MRIScans_Masks
# from writeout_dataset import Export2CompressedNifiti
import numpy as np
import os
import pydicom
import pandas as pd
import nibabel as nib

# cutoffSlice = 400
# seg_scan_dir = 'D:/MRI - Tairawhiti (User POV)'
# folders = ['10_AutoBindWATER_450_13A']
# output_seg_scan_dir = ('{}/Patient Segmentation Tasks (Google Colab)').format(seg_scan_dir)

def preprocessTestScans(cutoffSlice, seg_scan_dir, folders):
    output_seg_scan_dir = ('{}/Patient Segmentation Tasks (Google Colab)').format(seg_scan_dir)
    def read_dicom_files(directory):
        dicom_files = []
        files = os.listdir(directory)
        sorted_files = sorted(files)
        for filename in sorted_files:
            filepath = os.path.join(directory, filename)
            if os.path.isfile(filepath) and filename.endswith('.dcm'):
                try:
                    dicom_file = pydicom.dcmread(filepath)
                    dicom_files.append(dicom_file)
                except pydicom.errors.InvalidDicomError:
                    print(f"Skipping file: {filename}. It is not a valid DICOM file.")
        return dicom_files

    def flatten_2d_array(arr):
        flattened = []
        for row in arr:
            flattened.extend(row)
        return flattened

    def ReadIn_MRIScans_Masks(scans_path, folders):
        print('Patient Scan Data: ', folders)
        scan_pixel_data = []
        scan_coordinate_data = []
        single_scan_pixel_data = []
        scan_coordinate_data = []
        scan_orientation_data = []
        scan_pixelspacing_data = []


        single_paitent_scans_path =  scans_path + ('/Raw DICOM MRI Scans/{}').format(folders)
        dicom_files = read_dicom_files(single_paitent_scans_path)

        # Extracting pixel data
        for i in range (len(dicom_files)):
            single_scan_pixel_data.append(dicom_files[i].pixel_array)
        scan_pixel_data.append(single_scan_pixel_data)

        training_scans = flatten_2d_array(scan_pixel_data)
        training_scans = np.array(training_scans)
    
        # Coordinate Data
        single_paitent_scans_path =  scans_path + '/{}'.format(folders)
        for i in range (len(dicom_files)):
            scan_coordinate_data.append(dicom_files[i].ImagePositionPatient)
            scan_orientation_data.append(dicom_files[i].ImageOrientationPatient)
            scan_pixelspacing_data.append(dicom_files[i].PixelSpacing)
            
        coord_data = pd.DataFrame(scan_coordinate_data, columns=["x", "y", "z"])
        return training_scans

    def Export2CompressedNifiti(imgs_train, scans_path, colab_fname):
        nii_img_train = nib.Nifti1Image(imgs_train, affine=np.eye(4))
        output_file_path = ('{}/msk_{}.nii.gz').format(scans_path, colab_fname)
        nib.save(nii_img_train, output_file_path)

        print("Conversion to NIfTI complete with shape: ", nii_img_train.shape)
        print(('Successfully Exported Preprocessed Data! -> {}').format(output_file_path))
        print('\n')


    for patient in folders:
        patient_scan_data = ReadIn_MRIScans_Masks(seg_scan_dir, patient)

        patient_scan_data = patient_scan_data[(int(cutoffSlice)):-1]

        patient_scan_data = np.array(patient_scan_data)
        patient_scan_data = patient_scan_data.astype('float32')
        patient_scan_data /= 255.  # scale scans to [0, 1]

        patient_scan_data = np.expand_dims(patient_scan_data, axis=-1)
        print('Prediction Input Scan Shape: ', patient_scan_data.shape)

        
        scan_idx_fnamecolab = (patient.split('_'))[0]
        scan_idx_fnamecolab = '{:03d}'.format(int(scan_idx_fnamecolab))
        Export2CompressedNifiti(patient_scan_data, output_seg_scan_dir, scan_idx_fnamecolab)