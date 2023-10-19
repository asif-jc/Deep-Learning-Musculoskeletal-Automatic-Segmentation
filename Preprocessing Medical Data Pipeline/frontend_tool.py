import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import main  # Import your Python script with the desired function

def create_subtitle(text):
    subtitle_label = tk.Label(root, text=text, font=("Helvetica", 12, "bold"), pady=5)
    subtitle_label.pack()


def run_script_preprocessing():
    ScanDirectory = entry_scan_directory.get()
    createFolders = parameter0_combobox.get()
    scan_folder_name = entry_parameter5.get()
    mask_file_name = entry_parameter6.get()
    multiclassSeg = parameter2_combobox.get()
    imagepreprocessing = parameter3_combobox.get()
    cropping_dimensions = entry_parameter4.get()

    main.Preprocessing(ScanDirectory, createFolders, scan_folder_name, mask_file_name, multiclassSeg, imagepreprocessing, cropping_dimensions)

def run_script_prep_test_scan():
    # messagebox.showinfo("Input the directory where the raw MRI scans are located, Example Input: D:/MRI - Tairawhiti")
    ScanDirectory = entry_scan_directory.get()
    createFolders = parameter0_combobox.get()
    scan_folder_name = entry_parameter5.get()
    mask_file_name = entry_parameter6.get()
    multiclassSeg = parameter2_combobox.get()
    imagepreprocessing = parameter3_combobox.get()
    cropping_dimensions = entry_parameter4.get()

    main.Preprocessing(ScanDirectory, createFolders, scan_folder_name, mask_file_name, multiclassSeg, imagepreprocessing, cropping_dimensions)


# Create the main window
root = tk.Tk()
root.title("Lower Limb Musculoskeletal Automatic Segmentation Tool")

create_subtitle("Preprocessing Stage!")

# Create input fields
label_parameter0 = tk.Label(root, text="Create Required Folders For Data Storage (ONLY RUN ONCE!):")
label_parameter0.pack()
parameter0_values = ["True", "False"]
parameter0_combobox = ttk.Combobox(root, values=parameter0_values)
parameter0_combobox.pack()

label_scan_directory = tk.Label(root, text="MRI DICOM Base Directory (Example: D:/MRI - Tairawhiti):")  # Changed label text
label_scan_directory.pack()
entry_scan_directory = tk.Entry(root)  # Changed variable name
entry_scan_directory.pack()

label_parameter5 = tk.Label(root, text="Scan Folder Name (Example: 6_AutoBindWATER_650_9B):") 
label_parameter5.pack()
entry_parameter5 = tk.Entry(root)
entry_parameter5.pack()

label_parameter6 = tk.Label(root, text="Mask File Name (File Type: _.nii) (Example: 6_RR_fibula_9B):") 
label_parameter6.pack()
entry_parameter6 = tk.Entry(root)
entry_parameter6.pack()

label_parameter2 = tk.Label(root, text="Apply Multi-Class Segmentation Preprocessing (Only set to TRUE when all desired musculoskeletal masks for a patient have been exported!):")
label_parameter2.pack()
parameter2_values = ["True", "False"]
parameter2_combobox = ttk.Combobox(root, values=parameter2_values)
parameter2_combobox.pack()

label_parameter3 = tk.Label(root, text="Apply Image Preprocessing Techniques:")
label_parameter3.pack()
parameter3_values = ["True", "False"]
parameter3_combobox = ttk.Combobox(root, values=parameter3_values)
parameter3_combobox.pack()

label_parameter4 = tk.Label(root, text="Cropping Dimensions (If Apply_Image_Preprocessing_Techniques = True) (Format: [top,bottom,left,right]):") 
label_parameter4.pack()
entry_parameter4 = tk.Entry(root)
entry_parameter4.pack()

# Create run button
run_button = tk.Button(root, text="Run Preprocessing Layer", command=run_script_preprocessing)
run_button.pack()

create_subtitle("Processing Patient Scan For Automatic Segmentation!")

label_parameter7 = tk.Label(root, text="Patient DICOM Scan Folder Names (Example: [6_AutoBindWATER_650_9B, 3_AutoBindWATER_650_4A]):") 
label_parameter7.pack()
entry_parameter7 = tk.Entry(root)
entry_parameter7.pack()

label_parameter8 = tk.Label(root, text="Approximate Slice Index of Pelvis (Lowest Slice Index of Any Region of Interest Being Segmented):") 
label_parameter8.pack()
entry_parameter8 = tk.Entry(root)
entry_parameter8.pack()

# Start the main loop
root.mainloop()
