# Automatic Segmentation for Lower Limb Bones & Muscles using Deep Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust deep learning-based pipeline for automatic segmentation of musculoskeletal structures from MRI scans, developed for the Tairawhiti paediatric imaging study. This tool enables efficient and accurate segmentation of the tibia, femur, fibula, and pelvis structures from MRI scans of children.

## Overview

The musculoskeletal structure of children remains a significantly underexplored domain in research, plagued by challenges in the analysis of medical imaging data. Within medical imaging workflows, segmentation plays a pivotal role by identifying and localizing anatomical structures, enabling the study of their morphological changes over time. Manual segmentation, especially in paediatric populations, is a laborious and time-intensive task that demands meticulous annotation by expert researchers.

This project addresses the challenge of automatic segmentation of lower limb musculoskeletal structures from MRI scans in paediatric populations. By leveraging transfer learning and advanced deep learning techniques, our solution outperforms state-of-the-art biomedical segmentation frameworks while maintaining high computational efficiency.

![image](https://github.com/asif-jc/Deep-Learning-Musculoskeletal-Automatic-Segmentation-/assets/126116359/469661d9-c3a0-45d1-8109-cb2372f06b6c)

## Features

- **Complete Segmentation Pipeline**: Data preprocessing, model training, inference, and post-processing
- **Transfer Learning Architecture**: ResNet34 U-Net model pre-trained on ImageNet
- **Multi-class Segmentation**: Simultaneously segments tibia, femur, fibula, and pelvis structures
- **User-friendly GUI**: Simple interface for preprocessing, segmentation, and visualization
- **Post-processing Algorithms**: Denoising and artifact removal for optimized segmentation results
- **High Performance**: Achieves 89% mean Dice Similarity Coefficient (DSC) across all bone groups
- **Model Transferability**: Demonstrates strong generalizability across different paediatric populations

## Repository Structure

```
├── AutoSegmentationGUI.py            # Main GUI application
├── Preprocessing Medical Data Pipeline/  # Data preprocessing components
│   ├── MSKMulticlass.py              # Multi-class segmentation mask creation
│   ├── preprocessing.py              # Core preprocessing functions
│   ├── data_augmentation.py          # Data augmentation utilities
│   ├── image_preprocessing.py        # Image processing functions
│   └── ...                           # Other preprocessing utilities
├── Deep Learning Segmentation Models/   # Model implementations
│   ├── 2D U-Net.ipynb                # Baseline 2D U-Net implementation
│   ├── ResNet34 U-Net.ipynb          # Transfer learning ResNet34 U-Net
│   └── nnUNet.ipynb                  # nnU-Net implementation and comparison
└── README.md                         # Project documentation
```

## Results

Our ResNet34 U-Net model demonstrates superior performance compared to baseline models and the state-of-the-art nnU-Net framework:

| Model | Mean DSC | Mean Volume Error (cm³) |
|-------|----------|-------------------------|
| ResNet34 U-Net | 0.89 | 8.4 |
| 2D U-Net (nnU-Net) | 0.85 | 13.7 |
| 3D U-Net (nnU-Net) | 0.82 | 14.8 |
| 2D U-Net | 0.73 | 23.9 |

Detailed performance across individual bone groups:

| Bone Structure | ResNet34 U-Net (DSC) |
|----------------|----------------------|
| Tibia | 0.93 ± 0.02 |
| Femur | 0.95 ± 0.01 |
| Fibula | 0.78 ± 0.04 |
| Pelvis | 0.88 ± 0.03 |

![image](https://github.com/asif-jc/Deep-Learning-Musculoskeletal-Automatic-Segmentation-/assets/126116359/516f6b75-d079-47ed-bac0-a4cdf73369bb)

## Installation

```bash
# Clone the repository
git clone https://github.com/username/Deep-Learning-Musculoskeletal-Automatic-Segmentation.git
cd Deep-Learning-Musculoskeletal-Automatic-Segmentation

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- Python 3.9+
- PyTorch 1.11+
- Keras 2.13+
- TensorFlow 2.13+
- SimpleITK 2.2+
- Nibabel 4.0+
- NumPy 1.22+
- PyDicom 2.4+
- PIL 9.3+
- scikit-learn 1.2+

## Usage

### Using the GUI

The easiest way to use the segmentation pipeline is through the provided GUI:

```bash
python AutoSegmentationGUI.py
```

The GUI allows you to:
1. Preprocess training data (DICOM MRI scans + NIFTI binary segmentation masks)
2. Preprocess inference/test data (raw DICOM MRI scans)
3. Run the automatic segmentation model
4. Visualize 2D slices of scans and masks

![image](https://github.com/asif-jc/Deep-Learning-Musculoskeletal-Automatic-Segmentation-/assets/126116359/878f69b6-6914-468a-8d0a-1f30ead7aa86)

### Preprocessing Pipeline

The preprocessing pipeline handles:
- DICOM to NIFTI conversion
- Image normalization
- Multi-class mask creation
- Data augmentation
- Dataset-specific preprocessing

![image](https://github.com/asif-jc/Deep-Learning-Musculoskeletal-Automatic-Segmentation-/assets/126116359/c4557e3a-0a0f-49c3-993b-7be684fdb241)

### Segmentation Model

Our ResNet34 U-Net model uses the following hyperparameters:
- Learning rate: Dynamic schedule starting at 1e-4 with 1e-6 decay per epoch
- Loss function: Combination of Dice Loss and Categorical Focal Loss
- Optimizer: Adam
- Batch size: 16
- Activation function: Softmax

![image](https://github.com/asif-jc/Deep-Learning-Musculoskeletal-Automatic-Segmentation-/assets/126116359/62c0de86-46cf-4a32-8128-1d35a37c2c5c)

## Model Training

The transfer learning approach uses:
1. ResNet34 backbone pre-trained on ImageNet
2. Customized U-Net decoder path
3. Skip connections to preserve spatial information
4. Early stopping based on validation loss convergence

## Citation

If you use this code in your research, please cite our work:

```bibtex
@article{cheena2023automatic,
  title={Automatic Segmentation for Lower Limb Bones \& Muscles using Deep Learning},
  author={Cheena, Asif Juzar and Rao, Pranav and Choisne, Julie},
  year={2023},
  publisher={University of Auckland}
}
```

## Acknowledgements

This research was conducted as part of the ENGSCI700: Part IV Project at the University of Auckland, supervised by Dr. Julie Choisne. The work contributes to the ongoing Tairawhiti paediatric imaging and computer modeling initiative in New Zealand.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For questions or collaborations, please contact:
- Asif Juzar Cheena
- Pranav Rao
