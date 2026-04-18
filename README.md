# 🎭 EmotionAI-Lite: High-Efficiency CNN for Facial Expression Recognition

**EmotionAI-Lite** is a streamlined Deep Learning solution designed to classify human emotions from facial imagery. Built on the **FER-2013 dataset**, this project implements a **Lightweight Convolutional Neural Network (Light-CNN)** architecture, optimizing the balance between predictive accuracy and computational efficiency.

---

## 🔬 Architectural Design

The core of this project is a custom-engineered Light-CNN optimized for 48x48 grayscale inputs. Unlike bloated models, **EmotionAI-Lite** focuses on feature extraction through:

* **Optimized Convolutional Blocks:** Three stages of Convolutions with **Batch Normalization** to stabilize learning and accelerate convergence.
* **Dimensionality Reduction:** Strategic use of **Max Pooling** and **Global Average Pooling (GAP)** to minimize parameters and prevent overfitting.
* **Regularization:** Integrated **Dropout** layers to ensure robust generalization across the 7 emotion categories (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral).

---

## 🛠️ Technical Stack & Pipeline

| Stage | Process |
| :--- | :--- |
| **Framework** | TensorFlow / Keras |
| **Data Handling** | NumPy & Pandas (One-Hot Encoding & Normalization) |
| **Optimization** | Adam Optimizer |
| **Loss Function** | Categorical Crossentropy |
| **Metrics** | Precision, Recall, and Accuracy |

---

## 🚀 Training & Optimization Strategy

To achieve a production-ready model, the pipeline utilizes advanced training callbacks:

* **EarlyStopping:** Monitored validation loss to halt training at the optimal peak, preventing over-optimization.
* **ModelCheckpoint:** Automatically serializes the best performing weights as `best_fer_lightcnn.h5`.
* **Data Partitioning:** Strict separation of Training, Validation, and Test sets based on the original `Usage` metadata to ensure unbiased evaluation.
