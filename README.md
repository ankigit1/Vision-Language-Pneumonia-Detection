# 🩻 Vision–Language Model Guided Framework for Pneumonia Detection Using LBP-Stacked Bilateral Chest X-Rays

This repository contains the implementation of a **Vision–Language Model (VLM) guided Siamese framework** for pneumonia detection using **Local Binary Pattern (LBP)**-enhanced, **bilaterally split chest X-rays**.  
The framework integrates *symmetry reasoning*, *texture encoding*, and *contrastive laterality supervision* to achieve robust and interpretable classification.

---

## 📘 Overview

Traditional deep CNN models often overlook *bilateral asymmetry*—a key diagnostic cue in pneumonia.  
Our proposed architecture addresses this gap by combining:

- **LBP texture features** — capturing fine structural details from X-ray textures.  
- **Siamese bilateral design** — explicitly comparing left vs right lung representations.  
- **Vision–Language model (VLM) guidance** — providing text-based “laterality consistency” supervision using prompt scores (e.g., *“left lung opacity”*, *“right lung opacity”*).  

This results in improved detection performance and interpretability, verified via Grad-CAM and laterality heatmaps.

---

## 🧠 Model Architecture

Input X-ray
├── Left Lung (LBP + CNN Encoder)
├── Right Lung (LBP + CNN Encoder)
└── Cosine & Feature Fusion → Classifier (Normal / Pneumonia)
↑
Vision–Language Guidance (Laterality Loss λ)


The total loss function is:
\[
\mathcal{L}_{\text{total}} = \mathcal{L}_{\cos} + \lambda \, \mathcal{L}_{\text{lat}}
\]

---

## 📊 Datasets Used

### **Primary Training & Validation**
**Kermany et al. (2018)** — *“Labeled Optical Coherence Tomography (OCT) and Chest X-Ray Images for Classification”*, Mendeley Data, V2,  
DOI: [10.17632/rscbjbr9sj.2](https://doi.org/10.17632/rscbjbr9sj.2)

- 5,863 pediatric chest X-rays  
- Classes: `NORMAL`, `PNEUMONIA`  
- Used for: training, validation, and primary testing  

### **External Generalization Testing**
**Sugianto, Dwi (2025)** — *“Chest X-Ray”*, Mendeley Data, V1,  
DOI: [10.17632/p5rm59k7ph.1](https://doi.org/10.17632/p5rm59k7ph.1)

- 3 balanced categories (Normal / Lung Opacity / Viral Pneumonia)  
- Used only for cross-dataset generalization evaluation

---

## 🧩 Repository Structure

Vision-Language-Pneumonia-Detection/
│
├── models/
│ └── siamese_densenet_lbp.py # Model architecture (LBP + Siamese + VLM)
│
├── utils/
│ ├── dataset_loader.py # Custom Dataset + transforms
│ ├── training_utils.py # Train / evaluate loops + metrics
│ ├── xai_utils.py # Grad-CAM visualization helpers
│
├── checkpoints/
│ └── LBP_Siamese_lambda0.25.pt # Trained model weights
│
├── results/
│ ├── confusion_matrices/ # Accuracy and F1 visualizations
│ └── gradcam_examples/ # Grad-CAM interpretability maps
│
├── main_train.py # Training script
├── main_test.py # Evaluation & confusion matrix
├── main_xai.py # Grad-CAM + laterality visualization
│
├── requirements.txt # Dependencies
└── README.md # Project documentation



---

## ⚙️ Installation

```bash
# Clone repository
git clone https://github.com/<your-username>/Vision-Language-Pneumonia-Detection.git
cd Vision-Language-Pneumonia-Detection

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt


License

This work is released under the MIT License.
You may use and modify the code for research and educational purposes with proper citation.

Authors

Ankit Garg, M.Tech. (Data Analytics) – NIT Jalandhar
Dr. Rabi Shaw, Assistant Professor, Department of IT, NIT Jalandhar

Contact

For academic collaborations or questions:
📧 ankitg.da.24@nitj.ac.in