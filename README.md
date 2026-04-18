# Breast Tumor Mammography Detection

A deep learning project using **FastAI** and **transfer learning** to classify mammography images into two categories: **normal** and **tumor**.

---

## Project Structure

```
Breast/
├── breast-tumor-mammography.ipynb   # Training and evaluation notebook
├── model.pkl                        # Exported trained model (FastAI)
└── README.md                        # Project documentation
```

---

## Dataset

- **Source:** [Breast Cancer Detection](https://www.kaggle.com/datasets/hayder17/breast-cancer-detection) on Kaggle
- **Dataset path (Kaggle):** `/kaggle/input/datasets/hayder17/breast-cancer-detection/`
- **Folder structure:**
  - `0/` → Normal images
  - `1/` → Tumor images

### Class Distribution

| Label     | Number of Images |
| --------- | :--------------: |
| Normal    |      2,225       |
| Tumor     |      1,158       |
| **Total** |    **3,383**     |

> The dataset is imbalanced — the "normal" class has roughly twice as many samples as the "tumor" class.

---

## Tech Stack

| Tool / Library  | Purpose                                |
| --------------- | -------------------------------------- |
| Python 3.12     | Programming language                   |
| FastAI          | High-level deep learning framework     |
| PyTorch         | Backend for model training             |
| Matplotlib      | Image visualization and plotting       |
| Kaggle (T4 GPU) | Training environment (NVIDIA Tesla T4) |

---

## Model Architecture

The model leverages **transfer learning** with a backbone pre-trained on ImageNet, fine-tuned on the mammography dataset.

Because the dataset is imbalanced and medical diagnostics require minimizing **False Negatives** (failing to detect an actual tumor), **class weights** were increased for the minority class (tumor) during training to penalize the model more heavily for missing positive cases.

## Getting Started

### Requirements

- Python 3.8+
- FastAI 2.x
- PyTorch (compatible with FastAI version)
- GPU recommended

### Install Dependencies

```bash
pip install fastai
```

### Download Dataset

```bash
kaggle datasets download hayder17/breast-cancer-detection
unzip breast-cancer-detection.zip -d data/
```

### Run the Notebook

Open `breast-tumor-mammography.ipynb` in Kaggle (recommended for GPU access) or a local Jupyter environment with GPU support, then run all cells.

### Use the Pre-trained Model

```python
from fastai.vision.all import load_learner

learn = load_learner('model.pkl')
pred, pred_idx, probs = learn.predict('path/to/image.jpg')

print(f"Prediction : {pred}")
print(f"Confidence : {probs[pred_idx]:.4f}")
```

---

## Results

> The model achieved a **more than 90% tumor detection rate**.
> Detailed training results (accuracy, loss curves, confusion matrix) are available inside the notebook `breast-tumor-mammography.ipynb`.

---

## Notes

- The model was trained on **Kaggle**
- This is an **academic project** and should **not** be used for real-world clinical diagnosis without proper clinical validation.

---

## License

This project is intended for **academic purposes only**.
