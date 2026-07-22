# Student Placement Predictor (Part 1) 🎓

A Machine Learning project and interactive Streamlit web application that predicts student job placement status based on **Intelligence Quotient (IQ)** and **Cumulative Grade Point Average (CGPA)** using a trained Logistic Regression model.

---

## 📁 Repository Structure

```
├── app.py                      # Streamlit Web Application
├── model.pkl                   # Trained Logistic Regression Model
├── scaler.pkl                  # Fitted StandardScaler object
├── placement_dataset (1).csv   # Training Dataset
├── Untitled.ipynb              # Model Training & EDA Notebook
├── requirements.txt            # Python Dependencies
└── README.md                   # Project Documentation
```

---

## ⚙️ Model Details

- **Model Type**: Logistic Regression (`sklearn.linear_model.LogisticRegression`)
- **Features Used**: `iq`, `cgpa`
- **Feature Preprocessing**: `StandardScaler`
- **Target Variable**: `placement` (0 = Not Placed, 1 = Placed)

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/RajSingh2006-git/Student_Placement-Part_1.git
cd Student_Placement-Part_1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ✨ Application Features

- **Input Fields**: Easily enter student's IQ and CGPA.
- **Instant Prediction**: Predicts placement status with calculated probability score.
- **Animations & Motivational Messages**:
  - **Placed**: Celebratory balloon animations 🎉
  - **Not Placed**: Motivational skill-improvement guidance 💪
