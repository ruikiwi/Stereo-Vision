## **Stereo Vision Project**

This project explores stereo correspondence, focusing on the implementation and comparison of two algorithms: Block Matching and Graph Cuts. Stereo correspondence is crucial in computer vision for reconstructing 3D models from stereo image pairs by finding pixel-to-pixel correspondence.

### **Methods**
Block Matching: Uses a sliding window and cost functions like Sum of Squared Differences (SSD) to compute disparity.

Graph Cuts: Utilizes energy minimization techniques, specifically Markov Random Fields (MRF) and Alpha-Expansion, to find optimal disparity values.

### **Datasets**
Middlebury 2014 Stereo Datasets with Ground Truth: "Piano," "Backpack," "Umbrella," "Flowers."

### **Setup**
Set up the environment in conda:
> conda env create -f project.yml

Activate the environment:
> conda activate cv_proj

### **Results**




References
Fast Approximate Energy Minimization via Graph Cuts
Middlebury 2014 Stereo Datasets
