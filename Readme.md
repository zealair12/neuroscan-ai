# NeuroScan AI

**NeuroScan AI** is a web-based diagnostic assistant designed for medical professionals to analyze brain MRI scans using an AI-powered model. The system allows users to upload MRI images and receive instant classification results to assist in preliminary diagnosis and triage.

---

## 🚀 Features

- 📤 Upload MRI images (JPG, PNG)
- ⚙️ Preprocessing pipeline to match model input
- 🤖 Real-time inference with a trained image classifier (CNN-based)
- 🧾 Prediction display with confidence score
- 🎨 Custom Streamlit interface with sidebar instructions and doctor-friendly UI
- 🛠️ Modular backend built with Flask

---

## 🏗️ Project Structure

```
neuroscan-ai/
├── app/
│   ├── main.py                # Flask API backend
│   ├── model/                 # Placeholder for .h5 model
│   └── utils/
│       └── preprocess.py      # Image preprocessing logic
├── frontend/
│   └── streamlit_app.py       # Streamlit frontend UI
├── requirements.txt          # Python dependencies
└── .gitignore
```

---

## 🔧 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/zealair12/neuroscan-ai.git
cd neuroscan-ai
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv env
./env/Scripts/activate     # On Windows
```

### 3. Install Requirements
```bash
pip install -r requirements.txt
```

### 4. Add Your Model
Place a working `.h5` image classification model in `app/model/`. Example:
```
app/model/brain_tumor_classifier.h5
```

Make sure the model expects input in shape `(224, 224, 3)`.

### 5. Run Backend
```bash
python -m app.main
```

### 6. Run Frontend (in new terminal)
```bash
cd frontend
streamlit run streamlit_app.py
```

Then visit: `http://localhost:8501`

---

## ⚠️ Known Issues

- The current model predicts class `0` ("Glioma") too frequently regardless of input.
- We are actively replacing the model with a more accurate pre-trained one.

---

## 🧠 Class Labels (Expected)

```python
["Glioma", "Meningioma", "Pituitary", "No Tumor"]
```

---

## 📌 To Do

- [ ] Replace model with one from Hugging Face or Kaggle
- [ ] Add top-N predictions or visual class probabilities
- [ ] Option to export predictions as PDF
- [ ] Slack/Telegram integration for team notifications
- [ ] Deploy to Render or Hugging Face Spaces

