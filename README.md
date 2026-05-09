
# 🩺 MedVision AI
### *Multimodal Tuberculosis Diagnostic Assistant*

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)

**MedVision AI** is a decision-support tool designed to assist radiologists in identifying Tuberculosis (TB) by fusing computer vision with clinical patient data. By utilizing **Explainable AI (XAI)**, the system provides transparency into the neural network's decision-making process.

---

## 🧠 The Architecture
The system employs a **Multimodal Fusion Network**:
1.  **Vision Branch:** A pre-trained **EfficientNetV2** backbone extracts deep spatial features from Chest X-rays.
2.  **Tabular Branch:** A Multi-Layer Perceptron (MLP) processes patient vitals (Age, Systolic BP, Glucose).
3.  **Fusion Layer:** Features from both branches are concatenated and passed through a final classifier to determine the TB risk probability.

---

## 🔍 Explainable AI (Grad-CAM)
To prevent the "Black Box" problem in healthcare, this tool includes a **Neural Heatmap** generator. 
*   **Red Zones:** Indicate high-intensity areas where the model identified pathological patterns (e.g., infiltrates or cavities).
*   **Blue Zones:** Areas the model considered normal or irrelevant to the diagnosis.

---

## 📊 Technical Highlights
| Feature | Implementation |
| :--- | :--- |
| **Model** | EfficientNetV2 (Fused with Tabular MLP) |
| **Input** | Chest X-ray (224x224) + 3 Clinical Vitals |
| **Optimization** | Binary Cross Entropy Loss + Adam Optimizer |
| **Visualization** | Grad-CAM (Gradient-weighted Class Activation Mapping) |
| **UI/UX** | Dark-themed Medical Dashboard (Streamlit) |

---

### **📈 System Preview**
<img width="1365" height="945" alt="localhost_8501_" src="https://github.com/user-attachments/assets/885f9545-75b6-48b4-b765-aba1ee39a783" />

<img width="1365" height="945" alt="localhost_8501_ (1)" src="https://github.com/user-attachments/assets/964ddf32-ebdc-4b0d-8fda-c34964187349" />


⚠️ Medical Disclaimer
This application is for educational and research purposes only. It is not intended to be a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified healthcare provider.
