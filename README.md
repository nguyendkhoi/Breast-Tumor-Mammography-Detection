# 🔬 Breast Tumor Mammography Detection

A deep learning project using **FastAI** and **transfer learning** to classify mammography images into two categories: **normal** and **tumor**.

---

## 📁 Project Structure

```
Breast/
├── breast-tumor-mammography.ipynb   # Training and evaluation notebook
├── model.pkl                        # Exported trained model (FastAI)
└── README.md                        # Project documentation (this file)
```

---

## 📊 Dataset

- **Source:** [Breast Cancer Detection](https://www.kaggle.com/datasets/hayder17/breast-cancer-detection) on Kaggle
- **Dataset path (Kaggle):** `/kaggle/input/datasets/hayder17/breast-cancer-detection/`
- **Folder structure:**
  - `0/` → Normal images
  - `1/` → Tumor images

### Class Distribution

| Label   | Number of Images |
|---------|:----------------:|
| Normal  | 2,225            |
| Tumor   | 1,158            |
| **Total** | **3,383**      |

> ⚠️ The dataset is imbalanced — the "normal" class has roughly twice as many samples as the "tumor" class.

---

## Motivation: Reducing False Negatives

In a medical screening context, a **false negative** — predicting "normal" when the patient actually has a tumor — is far more dangerous than a false positive. Missing a real tumor can delay treatment and worsen patient outcomes.

To address this, the model applies **higher class weight to the tumor class** during training. This penalizes the model more heavily for missing tumors, shifting its decision boundary to be more sensitive (higher recall) on the positive class, at the cost of a slightly higher false positive rate.

```python
# Example: applying class weights in FastAI
learn = vision_learner(dls, resnet34, loss_func=CrossEntropyLossFlat(weight=tensor([1.0, 2.0])))
```

> The weight values can be tuned — a higher weight on the tumor class increases recall at the expense of precision.

---

## 🛠️ Tech Stack

| Tool / Library | Purpose                                 |
|----------------|-----------------------------------------|
| Python 3.12    | Programming language                    |
| FastAI         | High-level deep learning framework      |
| PyTorch        | Backend for model training              |
| Matplotlib     | Image visualization and plotting        |
| Kaggle (T4 GPU)| Training environment (NVIDIA Tesla T4)  |

---

## 🧠 Model Architecture

The model leverages **transfer learning** with a backbone pre-trained on ImageNet, fine-tuned on the mammography dataset.

### DataBlock Configuration

```python
dbl = DataBlock(
    blocks=(ImageBlock, CategoryBlock),
    splitter=GrandparentSplitter(),
    get_items=get_image_files,
    get_y=get_label,
    item_tfms=Resize(512, method=ResizeMethod.Pad),
    batch_tfms=[
        *aug_transforms(
            mult=2,
            do_flip=True,
            flip_vert=True,
            max_rotate=13.0,
            min_zoom=0.90,
            max_zoom=1.1
        ),
        Normalize.from_stats(*imagenet_stats),
    ]
)
```

### Key Configuration Details

| Parameter | Value / Description |
|-----------|---------------------|
| Input image size | 512×512 (padded to preserve aspect ratio) |
| Batch size | 16 (small batch for better generalization) |
| Horizontal flip | ✅ Enabled |
| Vertical flip | ✅ Enabled |
| Max rotation | ±13° |
| Zoom range | 90% – 110% |
| Augmentation multiplier | ×2 |
| Normalization | ImageNet statistics |
| Data split strategy | `GrandparentSplitter` (folder-based) |

---

## ▶️ Getting Started

### Requirements

- Python 3.8+
- FastAI 2.x
- PyTorch (compatible with FastAI version)
- GPU recommended (tested on NVIDIA Tesla T4)

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

## 📈 Results

### Key Metrics

| Metric | Value |
|--------|:-----:|
| Tumor Detection Rate | **70%** |

- **Reducing False Negatives**: In medical screening, missing a tumor is highly dangerous. Driven by the goal of reducing the probability of **false negatives**, we **increased the class weight** for the tumor class during model training.
- **Outcome**: As a result of this weighting strategy, the rate of correctly predicting that a person has a breast tumor (and it actually being a breast tumor) is **70%**.

> Detailed results including accuracy, loss curves, and the full confusion matrix are available inside the notebook `breast-tumor-mammography.ipynb`.

---

## 📌 Notes

- The model was trained on **Kaggle** using an **NVIDIA Tesla T4** GPU.
- The dataset is **imbalanced**. Consider applying techniques such as class weights, oversampling (SMOTE), or focal loss to improve performance on the minority class (tumor).
- This is an **academic / research project** and should **not** be used for real-world clinical diagnosis without proper clinical validation.

---

## 📄 License

This project is intended for **academic and research purposes only**.
