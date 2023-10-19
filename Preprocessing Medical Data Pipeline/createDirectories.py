import os
import shutil

def createDirectoriesfunc():
    def make_if_dont_exist(folder_path,overwrite):
        """
        creates a folder if it does not exists
        input:
        folder_path : relative path of the folder which needs to be created
        over_write :(default: False) if True overwrite the existing folder
        """
        if os.path.exists(folder_path):

            if not overwrite:
                print(f"{folder_path} exists.")
            else:
                print(f"{folder_path} overwritten")
                shutil.rmtree(folder_path)
                os.makedirs(folder_path)

        else:
            os.makedirs(folder_path)
            print(f"{folder_path} created!")



    base_directory = 'D:/MRI - Tairawhiti (User POV)'
    new_folder_name = ['nnUNet Data', 'Raw NIFITI Segmentation Masks (3D Slicer Output)', 'Pre-Trained Models (Google Colab)', 'Model Code', 'Raw DICOM MRI Scans', 'Patient Segmentation Tasks (Google Colab)']
    overwrite = False
    for new_fname in new_folder_name:
        folder_path = os.path.join(base_directory, new_fname)
        make_if_dont_exist(folder_path, overwrite)


    base_directory = 'D:/MRI - Tairawhiti (User POV)/nnUNet Data'
    new_folder_name = ['masks', 'multiclass_masks', 'scans']
    for new_fname in new_folder_name:
        folder_path = os.path.join(base_directory, new_fname)
        make_if_dont_exist(folder_path, overwrite)


    base_directory = 'D:/MRI - Tairawhiti (User POV)/nnUNet Data/masks'
    new_folder_name = ['FEMUR', 'FIBULA', 'TIBIA', 'PELVIS']
    for new_fname in new_folder_name:
        folder_path = os.path.join(base_directory, new_fname)
        make_if_dont_exist(folder_path, overwrite)
