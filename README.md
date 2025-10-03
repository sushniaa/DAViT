# DAViT: Domain Adaptive Vision Transformer for Robust Skin Lesion Multi-Classification with Domain Penalization
![License](https://img.shields.io/badge/License-MIT-blue.svg) 
![Python](https://img.shields.io/badge/Python-3.10+-blue.svg) 
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-brightgreen.svg)

## 📖 Overview
This repository contains the official implementation of **DAViT (Domain Adaptive Vision Transformer)**, a novel deep learning framework designed to enhance **robustness and generalization** in **skin lesion multi-classification** tasks.

DAViT addresses key challenges in medical imaging:
- Class imbalance  
- Anatomical domain shifts (e.g., torso, head/neck, palms/soles)  
- Lack of interpretability, which limits clinical adoption  

The **core innovations** include:
- **Fixed domain embeddings**: Injecting anatomical site metadata into the Vision Transformer’s attention mechanism.  
- **Dynamic domain penalization**: A fairness-aware strategy to balance losses across underrepresented domains.  

**Evaluation**:  
On a curated dataset of **57,299 dermoscopic images** from ISIC 2019 & 2020 challenges, DAViT achieves:  
- **Accuracy**: 0.9044  
- **AUC**: 0.9846  
- Stronger performance on **rare lesion types**.  

The codebase is **research-oriented**, emphasizing reproducibility, ablations (DAT-Learned vs. ViT-Baseline), and **explainable AI (XAI)** via HiResCAM & LayerCAM for clinical trust.  
---

## ✨ Features
- **Domain Penalization**: Dynamic loss balancing across anatomical domains, improving fairness & robustness.  
- **Domain-Adaptive ViT**: Extends ViT-Base with fixed or learnable domain embeddings.  
- **Model Variants**:
  - **DAT-Fixed** → Main DAViT model with fixed embeddings  
  - **DAT-Learned** → Ablation with dynamic embeddings  
  - **ViT-Baseline** → Standard ViT without domain adaptation  
- **Loss Functions**:  
  - Focal Loss (γ=3) → Handles class imbalance  
  - Domain penalization → Balances underrepresented domains  
- **Evaluation Metrics**: Accuracy, Balanced Accuracy, AUC, Sensitivity, Specificity, Dice coefficient, ROC curves, confusion matrices.  
- **XAI Integration**: HiResCAM & LayerCAM for interpretability and clinical validation.  
- **Dataset Curation**: Scripts for duplicate removal (PHash) and preprocessing.  
- **Reproducibility**: Jupyter notebooks, pre-trained checkpoints, and detailed logs.  

---

## 📂 Repository Structure
```bash
DAViT/
├── README.md # Project overview
├── LICENSE # MIT License
├── requirements.txt # Python dependencies
├── .gitignore # Ignore checkpoints, results, etc.
│
├── csv/
│ └── whole_data_no_duplicates_kaggle.csv # Curated dataset CSV
│
├── checkpoints/ # Pre-trained model checkpoints
│ ├── DAViT_BACC_NEW_18E_BEST_MODEL.ckpt # DAT-Fixed
│ ├── DAT_LEARNED_BACC_NEW_18E_BEST_MODEL.ckpt # DAT-Learned
│ └── VIT_BASELINE_BACC_NEW_18E_BEST_MODEL.ckpt # ViT-Baseline
│
├── notebooks/ # Jupyter notebooks
│ ├── davit_model.ipynb # DAT-Fixed demo
│ ├── dat_learned_model.ipynb # DAT-Learned demo
│ └── vit_baseline.ipynb # ViT-Baseline demo
│
├── src/
│ ├── curated_dataset_creation.py # Dataset preprocessing
│ ├── dataset.py # Dataset + transforms
│ ├── evaluator.py # Evaluation logic
│ ├── loss.py # Focal Loss + domain penalization
│ ├── model.py # DAViT architecture
│ ├── test.py # Model evaluation
│ ├── train.py # Training script
│ ├── utils.py # Helper utilities
│ └── xai.py # HiResCAM, LayerCAM
```

---

## 📊 Datasets
- **Source**: ISIC 2019 (25,331 images) + ISIC 2020 (33,126 images)  
- **Curated dataset**: 57,299 unique dermoscopic images after duplicate removal with **PHash**.  
- **Classes**: 9 diagnostic categories (e.g., MEL, NV, BCC)  
- **Domains**: 9 anatomical sites  

➡️ Download datasets from the **[ISIC Archive](https://www.isic-archive.com/)** and preprocess using:  
```bash
python src/curated_dataset_creation.py
```

## ⚙️ Installation
# Clone the repo
```bash
git clone https://github.com/<your-username>/DAViT.git
cd DAViT
```

# Install dependencies
```bash
pip install -r requirements.txt
```
Requires Python 3.10+ and a GPU (e.g., Tesla P100).


## 🚀 Usage
## Training

```bash
# Main DAViT (DAT-Fixed)
python src/train.py --model dat_fixed --csv csv/whole_data_no_duplicates_kaggle.csv --epochs 18 --lr 1e-4
```
```bash
# Ablation: DAT-Learned
python src/train.py --model dat_learned --csv csv/whole_data_no_duplicates_kaggle.csv --epochs 18 --lr 1e-4
```
```bash
# Ablation: ViT-Baseline
python src/train.py --model vit_baseline --csv csv/whole_data_no_duplicates_kaggle.csv --epochs 18 --lr 1e-4
```

## Testing
```bash
python src/test.py --model dat_fixed --checkpoint checkpoints/DAViT_BACC_NEW_18E_BEST_MODEL.ckpt
```

## XAI Analysis
```bash
python src/xai.py --model dat_fixed --checkpoint checkpoints/DAViT_BACC_NEW_18E_BEST_MODEL.ckpt --num_samples 5
```

## 📈 Results Summary
Test set (5,731 images) after 18 epochs

| Model | Accuracy | AUC | Balanced Acc. | Sensitivity | Dice Coeff. | Test Loss |
|---|---|---|---|---|---|---|
| **DAViT (DAT-Fixed)** | **0.9044** | **0.9846** | **0.8209** | **0.8209** | **0.8335** | **0.3445** |
| DAT-Learned | 0.8967 | 0.9833 | 0.8089 | 0.8089 | 0.8218 | 0.3565 |
| ViT-Baseline | 0.9010 | 0.9819 | 0.8134 | 0.8134 | 0.8257 | 0.3578 |


## 📜 Citation
If you use this repository in your research, please cite:
```bibtex
@article{rajendran2025davit,
  title={DAViT: A Generalized Deep Learning Framework for Robust Skin Lesion Multi-Classification with Domain Penalization},
  author={Rajendran, Sushmetha Sumathi and H, Kiranchandran and H, Vishal and A, Sasithradevi and Seemakurthy, Karthik and Poornachari, Prakash and M, Vijayalakshmi},
  journal={*Under Review*},
  year={2025}
}
```

## 📄 License

This project is licensed under the MIT License - see LICENSE for details.

## 🤝 Contributions
Contributions are welcome!
Open an issue for bugs or feature requests
Submit pull requests for improvements

## 📚 Acknowledgements
Supported by the IndiaAI Fellowship (MeitY, Govt. of India)
Thanks to the ISIC Archive, PyTorch, and TIMM communities

## 📬 Contact
Sushmetha Sumathi Rajendran – GitHub: sushniaa
 – 📧 sush7niaa@gmail.com

Kiranchandran H – GitHub: kiranchh08
 – 📧 kiranchandranh@gmail.com

Vishal H - Github: vishalh25
 - 📧 vishal.harindrakumar@gmail.com

