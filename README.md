# BISINDO Hand Landmark Recognition

A sign language recognition project for **BISINDO** (Bahasa Isyarat Indonesia / Indonesian Sign Language) letters A–Z using **MediaPipe Hands** for landmark extraction and **MLP (Multilayer Perceptron)** for classification, with model deployment to the web via **TensorFlow.js**.

---

## 📋 Description

This project consists of two main components:

1. **Data Collection** — A Python script for collecting hand landmark data in real-time via webcam using MediaPipe Hands
2. **Model Training** — A Google Colab notebook for training an MLP classification model for BISINDO gestures and converting it to TensorFlow.js format

### Key Features

- 🖐️ **Automatic single/dual hand mode detection** based on letter label
- 📐 **Two-stage normalization**: Translation → Scale normalization
- 📊 **Consistent 126-feature format** (63 per hand × 2 hands) for all labels
- 🧠 **MLP model** with `128 → 64 → 26` neuron architecture
- 🌐 **Export to TensorFlow.js** for web browser deployment
- 📈 **Comprehensive evaluation**: Classification report, confusion matrix, accuracy & loss graphs

---

## 📁 Project Structure

```
final/
├── data-collection/                  # Data collection module
│   ├── collect_landmarks.py          # Main data collection script via webcam
│   └── requirements.txt             # Dependencies for data collection
│
├── model/                            # Trained model outputs
│   ├── label_encoder.pkl             # Label encoder (sklearn)
│   ├── best_model_mlp_savedmodel.zip  # TensorFlow SavedModel
│   └── mlp_tfjs_model.zip            # TensorFlow.js model (for web deployment)
│
├── Training_BISINDO.ipynb            # Model training notebook (Google Colab)
├── training_bisindo.py               # Script version of the notebook
├── bisindo_landmarks_scaled.csv      # Main dataset (scale normalized landmarks)
├── train.csv                         # Training dataset (70%)
├── validation.csv                    # Validation dataset (15%)
├── test.csv                          # Testing dataset (15%)
├── confusion_matrix_mlp.png          # Confusion matrix visualization
├── training_validation_mlp.png       # Accuracy/loss training curves
├── requirements.txt                  # Dependencies for training (Colab environment)
└── README.md                         # Documentation (this file)
```

---

## 🏷️ Label Classification

| Mode       | Letter Labels                                    |
|-----------|--------------------------------------------------|
| One Hand  | C, E, I, J, L, O, R, U, V, Z                    |
| Two Hands | A, B, D, F, G, H, K, M, N, P, Q, S, T, W, X, Y |

---

## 📊 Dataset Format

The file `bisindo_landmarks_scaled.csv` contains **127 columns**:

| Column | Description |
|---|---|
| `label` | BISINDO alphabet letter (A–Z) |
| `left_0_x` ... `left_20_z` | 63 left hand landmark features (21 points × 3 coordinates) |
| `right_0_x` ... `right_20_z` | 63 right hand landmark features (21 points × 3 coordinates) |

> **Note:** For single-hand labels, features of the undetected hand are filled with `0.0`.

### Normalization Process

1. **Translation Normalization** — Each landmark is subtracted by the wrist position (point 0) so that the wrist becomes the origin `(0, 0, 0)`
2. **Scale Normalization** — The translated landmarks are divided by the Euclidean distance between the max and min values, making them invariant to hand size

### Dataset Split

| Split    | Proportion | File        |
|----------|------------|-------------|
| Training | 70%        | `train.csv` |
| Validation | 15%      | `validation.csv` |
| Testing  | 15%        | `test.csv`  |

The split is performed using **stratified sampling** (preserving the proportion of each label) with `random_state=42`.

---

## 🧠 Model Architecture

The model uses a **Multilayer Perceptron (MLP)** with the following architecture:

```
Input (126 features)
    │
Dense(128, ReLU)
    │
Dropout(0.25)
    │
Dense(64, ReLU)
    │
Dropout(0.2)
    │
Dense(26, Softmax)  →  Output (A-Z)
```

### Training Hyperparameters

| Parameter         | Value                                       |
|-------------------|---------------------------------------------|
| Optimizer         | Adam (lr = 0.001)                           |
| Loss Function     | Sparse Categorical Crossentropy             |
| Epochs            | 100 (max)                                   |
| Batch Size        | 32                                          |
| Early Stopping    | patience = 10, restore best weights         |
| ReduceLROnPlateau | factor = 0.5, patience = 5, min_lr = 1e-6   |

### Training Outputs

| File                           | Purpose                                  |
|--------------------------------|------------------------------------------|
| `best_model_mlp_savedmodel.zip` | TensorFlow SavedModel archive           |
| `mlp_tfjs_model.zip`           | TensorFlow.js model for browser use      |
| `label_encoder.pkl`            | Label to index mapping                   |
| `confusion_matrix_mlp.png`     | Confusion matrix visualization           |
| `training_validation_mlp.png`   | Training and validation curve chart      |

## 📈 Results

The latest training run produced the following results:

| Metric | Value |
|--------|-------|
| Train Loss | 0.0072 |
| Train Accuracy | 0.9973 |
| Validation Loss | 0.0277 |
| Validation Accuracy | 0.9915 |
| Test Loss | 0.0531 |
| Test Accuracy | 0.9932 |

The notebook also generates `confusion_matrix_mlp.png` and `training_validation_mlp.png`.

---

## ⚙️ Installation & Usage

### A. Data Collection (Local)

#### Prerequisites
- Python 3.8+
- Webcam

#### Setup & Run

```bash
# Clone the repository
git clone https://github.com/YeeIsRizka/handsson-machine-learning.git
cd handsson-machine-learning/data-collection

# Install dependencies
pip install -r requirements.txt

# Run the script
python collect_landmarks.py
```

#### Data Collection Controls

| Key | Function                                  |
|-----|-------------------------------------------|
| `s` | Start data collection (5-second countdown) |
| `q` | Exit the program                          |

#### Usage Flow

1. Run the script → select a letter label → enter the number of samples
2. Press `s` on the camera window → 5-second countdown
3. Data is automatically captured every **1 second**
4. Samples are only saved if the detected number of hands matches the label mode
5. Data is appended to `bisindo_landmarks_scaled.csv`

---

### B. Model Training (Google Colab)

1. Upload `Training_BISINDO.ipynb` to [Google Colab](https://colab.research.google.com/)
2. Upload `bisindo_landmarks_scaled.csv` to the Colab environment
3. Run all cells sequentially
4. Download the trained models and visual outputs from the `model/` folder and notebook runtime output

#### Training Outputs

| File                     | Format               | Purpose                              |
|--------------------------|----------------------|--------------------------------------|
| `label_encoder.pkl`      | Pickle               | Letter label ↔ numeric index mapping |
| `best_model_mlp_savedmodel.zip` | TensorFlow SavedModel | Inference in Python/TensorFlow |
| `mlp_tfjs_model.zip`     | TensorFlow.js        | Web browser deployment               |

---

## 🔧 Configuration

### Data Collection

| Parameter                  | Default | Description                                   |
|----------------------------|---------|-----------------------------------------------|
| `SAMPLE_INTERVAL`          | `1.0`   | Time interval between samples (seconds)       |
| `PREPARATION_TIME`         | `5`     | Preparation time before starting (seconds)    |
| `min_detection_confidence` | `0.7`   | Minimum hand detection confidence             |
| `min_tracking_confidence`  | `0.7`   | Minimum hand tracking confidence              |

---

## 🧪 End-to-End Pipeline

```
┌─────────────────────────────────────────────────────────┐
│                    DATA COLLECTION                       │
│                                                         │
│  Webcam → MediaPipe Hands → Extract 21 Landmarks/Hand  │
│        → Translation Norm → Scale Norm → CSV            │
└──────────────────────────┬──────────────────────────────┘
                           │
                    bisindo_landmarks_scaled.csv
                           │
┌──────────────────────────▼──────────────────────────────┐
│                    MODEL TRAINING                        │
│                                                         │
│  CSV → Stratified Split (70/15/15)                      │
│      → Label Encoding → MLP Training                   │
│      → Evaluation (Report + Confusion Matrix)           │
│      → Export (SavedModel + TensorFlow.js)              │
└─────────────────────────────────────────────────────────┘
```

---

## 📚 Dataset References

The dataset was collected independently via webcam. The following BISINDO datasets were used as references for study:

- [BISINDO Alphabet — Achmad Noer](https://www.kaggle.com/datasets/achmadnoer/alfabet-bisindo)
- [BISINDO Dataset — Yunita Ayu](https://www.kaggle.com/datasets/yunitayupratiwi/bisindo-dataset/data)
- [BISINDO Letter Dataset — Alfredo](https://www.kaggle.com/datasets/alfredolorentiars/bisindo-letter-dataset)
- [BISINDO Hand Sign Detection — Rhio Sutoyo](https://github.com/rhiosutoyo/Indonesian-Sign-Language-BISINDO-Hand-Sign-Detection-Dataset)
- [Sign Language BISINDO — Bonar Sitorus](https://www.kaggle.com/datasets/bonarsitorus/sign-language-bisindo)
- [BISINDO Final — Sifaq Einstein](https://www.kaggle.com/datasets/sifaqeinstein/bisindo-final)

---

## 🛠️ Tech Stack

| Technology             | Purpose                                    |
|------------------------|--------------------------------------------|
| Python                 | Primary programming language               |
| MediaPipe              | Hand detection & landmark extraction       |
| OpenCV                 | Webcam access & visualization              |
| NumPy                  | Normalization computation                  |
| TensorFlow / Keras     | MLP model training                         |
| Scikit-learn           | Label encoding, train-test split, evaluation |
| TensorFlow.js          | Model conversion for web deployment        |
| Matplotlib & Seaborn   | Visualization (graphs, confusion matrix)   |
| Google Colab           | Training environment                       |

---

## 📜 License

This project is licensed under the [MIT License](LICENSE).

Copyright (c) 2026 Rizka Alfadilla

---

## 🤝 Contributing

Contributions via *issues* and *pull requests* are welcome. Please create an issue first to discuss any proposed changes.
