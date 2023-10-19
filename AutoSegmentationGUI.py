import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import os
import sys

sys.path.append(os.path.join(os.getcwd(), 'Model Code'))
import main

def create_subtitle(text):
    subtitle_label = tk.Label(root, text=text, font=("Helvetica", 12, "bold"), pady=5)
    subtitle_label.pack()


def run_script_preprocessing():
    ScanDirectory = entry_scan_directory.get("1.0", "end-1c")
    # runTask = parameter0_combobox.get()
    scan_folder_name = entry_parameter5.get("1.0", "end-1c")
    # mask_file_name = entry_parameter6.get("1.0", "end-1c")
    resizing = parameter2_combobox.get()

    main.Preprocessing(ScanDirectory, scan_folder_name, resizing)

# def run_script_data_aug():
#     aug_fname = entry_parameter14.get("1.0", "end-1c")
#     num_augs = entry_parameter4.get()
#     ScanDirectory = entry_scan_directory.get("1.0", "end-1c")

#     main.DataAug(ScanDirectory, aug_fname, num_augs)

def run_script_prep_test_scan():
    # runTask = parameter9_combobox.get()
    cutoffSlice = entry_parameter8.get()
    ScanDirectory = entry_scan_directory.get("1.0", "end-1c")
    folders = entry_parameter7.get("1.0", "end-1c")

    main.preprocessTestScansMain(cutoffSlice, ScanDirectory, folders)

def run_script_visualise():
    # runTask = parameter9_combobox.get()
    cutoffSlice = entry_parameter12.get()
    mask_file_name = entry_parameter13.get("1.0", "end-1c")
    ScanDirectory = entry_scan_directory.get("1.0", "end-1c")
    main.VisualiseSlice(ScanDirectory, mask_file_name, cutoffSlice)

def run_script_model():
    # runTask = parameter9_combobox.get()
    subjectfname = entry_parameter21.get("1.0", "end-1c")
    ScanDirectory = entry_scan_directory.get("1.0", "end-1c")
    main.AutoSegModel(ScanDirectory, subjectfname)


# Create the main window
root = tk.Tk()
root.title("Lower Limb Musculoskeletal Automatic Segmentation Tool")


label_scan_directory = tk.Label(root, text="Base Directory (Example: D:/MRI - Tairawhiti):", font=("Helvetica", 12, "bold"))  # Changed label text
label_scan_directory.pack()
entry_scan_directory = tk.Text(root, height=1, width=100)
entry_scan_directory.pack()
# entry_scan_directory = tk.Entry(root)  # Changed variable name
# entry_scan_directory.pack()

create_subtitle('-'*160)
create_subtitle("Preprocessing Training Data (DICOM MRI Scans + NIFITI Binary Segmentation Masks)")
create_subtitle('-'*160)

# Create input fields
# label_parameter0 = tk.Label(root, text="Run Preprocessing Raw DICOM MRI Scans & Raw NIFITI Segmentation Masks Task:")
# label_parameter0.pack()
# parameter0_values = ["True", "False"]
# parameter0_combobox = ttk.Combobox(root, values=parameter0_values)
# parameter0_combobox.pack()


label_parameter5 = tk.Label(root, text="Scan Folder Name (Example: 6_AutoBindWATER_650_9B):") 
label_parameter5.pack()
entry_parameter5 = tk.Text(root, height=1, width=40)
entry_parameter5.pack()

# label_parameter6 = tk.Label(root, text="Mask File Name (File Type: _.nii) (Example: 6_RR_fibula_9B) (Note: if Multi-Class, set to 'multi'!):") 
# label_parameter6.pack()
# entry_parameter6 = tk.Text(root, height=1, width=40)
# entry_parameter6.pack()

label_parameter2 = tk.Label(root, text="Select Image Data Size: ")
label_parameter2.pack()
parameter2_values = ["512x512", "256x256"]
parameter2_combobox = ttk.Combobox(root, values=parameter2_values)
parameter2_combobox.pack()

# label_parameter3 = tk.Label(root, text="Apply Image Preprocessing Techniques:")
# label_parameter3.pack()
# parameter3_values = ["True", "False"]
# parameter3_combobox = ttk.Combobox(root, values=parameter3_values)
# parameter3_combobox.pack()

# label_parameter4 = tk.Label(root, text="Cropping Dimensions (If Apply_Image_Preprocessing_Techniques = True) (Format: [top,bottom,left,right]):") 
# label_parameter4.pack()
# entry_parameter4 = tk.Entry(root)
# entry_parameter4.pack()

# Create run button
create_subtitle('-'*80)
run_button = tk.Button(root, text="Run Preprocessing Layer (Scan Stack & Segmentation Masks)", command=run_script_preprocessing)
run_button.pack()

create_subtitle('-'*160)
create_subtitle("Preprocessing Inference/Test Data (Raw DICOM MRI Scans)")
create_subtitle('-'*160)

# label_parameter9 = tk.Label(root, text="Run Preprocessing Patient Scans For Automatic Segmentation Task:")
# label_parameter9.pack()
# parameter9_values = ["True", "False"]
# parameter9_combobox = ttk.Combobox(root, values=parameter3_values)
# parameter9_combobox.pack()


label_parameter7 = tk.Label(root, text="Scan Folder Name (Example: 6_AutoBindWATER_650_9B):") 
label_parameter7.pack()
entry_parameter7 = tk.Text(root, height=1, width=40)
entry_parameter7.pack()


label_parameter8 = tk.Label(root, text="Approximate Slice Index of Pelvis (Lowest Slice Index of Any Region of Interest Being Segmented):") 
label_parameter8.pack()
entry_parameter8 = tk.Entry(root)
entry_parameter8.pack()

create_subtitle('-'*80)
run_button = tk.Button(root, text="Generate Preprocessed Inference/Prediction (Scan Stack)", command=run_script_prep_test_scan)
run_button.pack()


create_subtitle('-'*160)
create_subtitle("Run Automatic Segmentation Model (Pre-trained resnet34 U-Net)")
create_subtitle('-'*160)

# label_parameter9 = tk.Label(root, text="Run Preprocessing Patient Scans For Automatic Segmentation Task:")
# label_parameter9.pack()
# parameter9_values = ["True", "False"]
# parameter9_combobox = ttk.Combobox(root, values=parameter3_values)
# parameter9_combobox.pack()


label_parameter21 = tk.Label(root, text="Subject Scan File Name (Example: msk_006):") 
label_parameter21.pack()
entry_parameter21 = tk.Text(root, height=1, width=40)
entry_parameter21.pack()

create_subtitle('-'*80)
run_button = tk.Button(root, text="Generate Segmentation Output", command=run_script_model)
run_button.pack()

# create_subtitle('-'*160)
# create_subtitle("Data Augmentation")
# create_subtitle('-'*160)


# label_parameter14 = tk.Label(root, text="Preprocessed Data File Name (Example: msk_006):") 
# label_parameter14.pack()
# entry_parameter14 = tk.Text(root, height=1, width=40)
# entry_parameter14.pack()

# label_parameter4 = tk.Label(root, text="Number of Augmentations Per Image:") 
# label_parameter4.pack()
# entry_parameter4 = tk.Entry(root)
# entry_parameter4.pack()


# create_subtitle('-'*80)
# run_button = tk.Button(root, text="Augment Training Data", command=run_script_data_aug)
# run_button.pack()



create_subtitle('-'*160)
create_subtitle("Visualisations of 2D Scans & Masks")
create_subtitle('-'*160)

label_parameter12 = tk.Label(root, text="Slice Number:") 
label_parameter12.pack()
entry_parameter12 = tk.Entry(root)
entry_parameter12.pack()

label_parameter13 = tk.Label(root, text="Mask File Name (Example: 4_R_tibia_5A, Example(Multi): msk_006) (Requires the preprocessed data!):") 
label_parameter13.pack()
entry_parameter13 = tk.Text(root, height=1, width=40)
entry_parameter13.pack()

create_subtitle('-'*80)
run_button = tk.Button(root, text="Generate Data Visualisations", command=run_script_visualise)
run_button.pack()
create_subtitle('-'*160)

# Start the main loop
root.mainloop()
