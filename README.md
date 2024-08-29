## **Stereo Vision Project**

This project explores stereo correspondence, focusing on the implementation and comparison of two algorithms in python: Block Matching and Graph Cuts. Stereo correspondence is crucial in computer vision for reconstructing 3D models from stereo image pairs by finding pixel-to-pixel correspondence.

### **Methods**
Block Matching: Uses a sliding window and cost functions like Sum of Squared Differences (SSD) to compute disparity.

Graph Cuts: Utilizes energy minimization techniques, specifically Markov Random Fields (MRF) and Alpha-Expansion, to find optimal disparity values.
The algorithm implemented is described in [Boykov et. al.'s graph cut method](https://doi.org/10.1109/34.969114).

### **Datasets**
[Middlebury 2014 Stereo Datasets with Ground Truth](https://vision.middlebury.edu/stereo/data/scenes2014/)

### **Setup**
Set up the environment in conda:
> conda env create -f project.yml

Activate the environment:
> conda activate cv_proj

### **Results**
![Result](https://github.com/user-attachments/assets/611c36c4-888a-4758-8ac7-cb875e89c899)
