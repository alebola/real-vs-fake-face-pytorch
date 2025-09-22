# 🧠 Real vs. Fake Face Classification (PyTorch)

This repo contains a **binary image classifier** for **real vs. fake faces** built in **PyTorch**.  
It includes:
- A clean **CNN baseline**.
- A **fine-tuned MobileNetV2** (ImageNet weights).
- **Stratified train/test split** (70/30).
- Proper **normalization** (dataset / ImageNet).
- Full evaluation: **Accuracy, Precision, Recall, F1** + **confusion matrix**.
- Handy visualization of **predictions**.

## 📂 Project Structure
```
notebooks/
real_fake_face_classifier.ipynb    # main notebook (CNN + MobileNetV2)
datasets/
real_and_fake_face/                # place your dataset here 
requirements.txt                   # pip deps
README.md
```

## 🗃️ Dataset
Place your images under:
```
datasets/real_and_fake_face/
real/   # real images
fake/   # fake images
```
The notebook uses `torchvision.datasets.ImageFolder`, so any folder structure with class subfolders works.

## 🧪 What’s inside
- **Baseline CNN** (small, didactic).
- **MobileNetV2 fine-tuning** with ImageNet normalization.
- **Stratified split** to keep class balance.
- **Augment only in train** (flip/rotation).
- **Evaluate on the whole test set** with rich metrics and confusion matrix.

## ▶️ How to run
1) Create/activate an environment and install deps:
```bash
pip install -r requirements.txt
```
2. Put the dataset in `datasets/real_and_fake_face/` (see structure above).
3. Open the notebook:
```bash
jupyter notebook notebooks/real_fake_face_classifier.ipynb
```
4. Run all cells. You’ll train:
- The CNN baseline.
- The MobileNetV2 with ImageNet weights and a small classifier head.

## 🔍 Metrics & Plots
The notebook prints:
- Accuracy / Precision / Recall / F1
- **Classification report** per class
- **Confusion matrix** (with nice formatting)
- Training curves for loss and accuracy
- Sample predictions grid (denormalized for visualization)

## 🛠️ Tech
- Python, PyTorch, Torchvision
- scikit-learn (metrics)
- Matplotlib / NumPy

## ⚠️ Notes
- The dataset is not included in the repository.
- For reproducibility, a fixed `SEED=42` is used for the split.
- Training is on CPU by default. If CUDA is available, the notebook uses it automatically.
