# 🌿 CropGuard AI — Plant Disease Detector

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-App-red?style=for-the-badge&logo=streamlit)
![MobileNetV2](https://img.shields.io/badge/Model-MobileNetV2-green?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**An AI-powered web application that detects plant diseases from leaf images in seconds.**
Built with MobileNetV2 transfer learning and deployed using Streamlit.

[📊 Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) · [🐛 Report Bug](../../issues)

</div>

---

## 📸 Screenshots

> <img width="1919" height="986" alt="Screenshot 1" src="https://github.com/user-attachments/assets/ca127503-fbb6-4c0b-8047-b12d066decd2" />
<img width="1919" height="991" alt="Screenshot 2" src="https://github.com/user-attachments/assets/bd72e7e9-e275-45e4-947a-914d8bd701a8" />



---

## ✨ Features

- 🔍 **Instant Disease Detection** — Upload a leaf photo and get results in seconds
- 🧠 **38 Disease Classes** across 14 crop types
- 📊 **Top 3 Predictions** with confidence scores
- 🟡 **Severity Assessment** — Healthy / Moderate / Severe
- 💊 **Treatment Advice** — Chemical, Organic & Prevention methods
- 📋 **Downloadable Report** for each analysis
- 🎨 **Clean Professional UI** built with custom CSS

---

## 🌱 Supported Crops & Diseases

| Crop | Diseases Detected |
|------|------------------|
| 🍅 Tomato | Early Blight, Late Blight, Leaf Mold, Mosaic Virus, Yellow Curl Virus + more |
| 🌽 Corn | Common Rust, Northern Leaf Blight, Gray Leaf Spot |
| 🥔 Potato | Early Blight, Late Blight |
| 🍇 Grape | Black Rot, Esca, Leaf Blight |
| 🍎 Apple | Apple Scab, Black Rot, Cedar Rust |
| 🍊 Orange | Citrus Greening (HLB) |
| + 8 more | Peach, Pepper, Blueberry, Raspberry, Soybean, Squash, Strawberry, Cherry |

---

## 🚀 How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/eyramana/crop-disease-detector.git
cd crop-disease-detector
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Add model files
Place these in a `models/` folder:
```
models/
├── crop_disease.tflite
└── class_names.json
```

### 4. Run the app
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` 🎉

---

## 📁 Project Structure

```
crop-disease-detector/
├── models/                      # TFLite model + class labels (not in repo)
│    ├── crop_disease.tflite
│    └── class_names.json
├── app.py                       # 🎯 Main Streamlit web app
├── train.py                     # Model training script
├── model.py                     # MobileNetV2 architecture
├── preprocess.py                # Image preprocessing & augmentation
├── evaluate.py                  # Model evaluation & metrics
├── requirements.txt             # Python dependencies
└── README.md
```

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **TensorFlow / TFLite** | Model training & optimized inference |
| **MobileNetV2** | Transfer learning from ImageNet |
| **Streamlit** | Interactive web application |
| **Pillow** | Image loading & processing |
| **NumPy** | Array operations |
| **PlantVillage Dataset** | 54,000+ training images |

---

## 🧠 Model Architecture

```
Input (224×224×3)
       ↓
MobileNetV2 (pretrained on ImageNet) ← Frozen layers
       ↓
GlobalAveragePooling2D
       ↓
Dense(256, ReLU) + Dropout(0.3)
       ↓
Dense(38, Softmax) ← 38 disease classes
```

- **Training:** Transfer learning — only top layers trained
- **Format:** Converted to TensorFlow Lite for fast inference
- **Input size:** 224 × 224 pixels

---

## 📊 How It Works

```
User uploads leaf image
         ↓
Image resized to 224×224
         ↓
TFLite model runs inference
         ↓
Top 3 predictions returned
         ↓
Disease info + treatment displayed
```

---

## 🔮 Future Improvements

- [ ] Real-time camera detection
- [ ] Mobile app version
- [ ] Multi-language support (Tamil, Hindi)
- [ ] Weather-based disease risk prediction
- [ ] More crop varieties

---

## 📄 Dataset

[PlantVillage Dataset](https://www.kaggle.com/datasets/emmarex/plantdisease) — 54,000+ leaf images across 38 plant disease categories.

---

## 👨‍💻 Author

**Ramana** — [@eyramana](https://github.com/eyramana)

> *"Built this project to help farmers identify crop diseases early and save their harvest using AI."*

---

## ⭐ If you found this useful, please give it a star!

<div align="center">
Made with ❤️ using Deep Learning & Streamlit
</div>

