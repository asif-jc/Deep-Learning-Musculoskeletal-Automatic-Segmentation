# Deep-Learning-Musculoskeletal-Automatic-Segmentation-
Honours Project: Automatic Segmentation for Lower Limb Bones &amp; Muscles using Deep Learning

The musculoskeletal structure of children remains a significantly underexplored domain in research, plagued by
challenges in the analysis of medical imaging data. Within medical imaging workflows, segmentation plays a
pivotal role by identifying and localizing anatomical structures, enabling the study of their morphological
changes over time. Manual segmentation, especially in paediatric populations, is a laborious and time-intensive
task that demands meticulous annotation by expert researchers. In this study, we propose a robust deep learning
based segmentation pipeline that performs automatic segmentation on Magnetic Resonance Imaging (MRI) on
populations of children. The segmentation model used within the study is the Resnet34 U-Net deep learning
architecture because of its strong segmentation accuracy and adaptability to transfer learning. The deep learning
segmentation pipeline consists of a medical data processing layer, the Resnet34 U-Net deep learning
segmentation model, and a denoising algorithm to post-process segmentation results. This is all integrated in a
user-friendly GUI for ease of use. Experimental results were compared to state-of-the-art deep learning based
segmentation methods within the biomedical domain, such as the nnU-Net framework, and the H-DenseUNet.
The experiment accuracy of the ResNet34 U-Net is 89% (DSC), and outperforms the nnU-Net framework’s 2D
U-Net and 3D U-Net models on the segmentation of the tibia, femur, fibula and pelvis musculoskeletal
structures. As well as the H-DenseUNet on the segmentation of the tibia structure.



![image](https://github.com/asif-jc/Deep-Learning-Musculoskeletal-Automatic-Segmentation-/assets/126116359/469661d9-c3a0-45d1-8109-cb2372f06b6c)

![image](https://github.com/asif-jc/Deep-Learning-Musculoskeletal-Automatic-Segmentation-/assets/126116359/c4557e3a-0a0f-49c3-993b-7be684fdb241)

![image](https://github.com/asif-jc/Deep-Learning-Musculoskeletal-Automatic-Segmentation-/assets/126116359/516f6b75-d079-47ed-bac0-a4cdf73369bb)

![image](https://github.com/asif-jc/Deep-Learning-Musculoskeletal-Automatic-Segmentation-/assets/126116359/878f69b6-6914-468a-8d0a-1f30ead7aa86)
