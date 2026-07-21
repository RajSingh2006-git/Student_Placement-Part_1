<<<<<<< HEAD
# Student_Placement-Part_1
=======
# 🎓 Student Placement Analytics & Machine Learning Predictor

An interactive Machine Learning web application built with **Streamlit**, **Scikit-Learn**, and **Plotly** to analyze student academic metrics (IQ & CGPA) and predict campus placement status.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-FF4B4B)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 📌 Project Overview

This repository contains the complete end-to-end Machine Learning pipeline and web application for student placement prediction:
1. **Jupyter Notebook (`Untitled.ipynb`):** Data exploration, duplicate handling, feature scaling, model training (`LogisticRegression`), and accuracy evaluation.
2. **Streamlit App (`app.py`):** Interactive dashboard featuring real-time predictions, decision boundary visualization, multi-model evaluation, and batch CSV processing.

---

## ✨ Key Features

- **🏠 Executive Overview & EDA:** Key metrics, data preview, summary statistics, and automated duplicate cleaning report.
- **📊 Interactive Visual Analytics:** Plotly scatter plots, feature distribution histograms, and correlation matrix heatmaps.
- **⚙️ Multi-Model Machine Learning:** Train and compare **Logistic Regression**, **Random Forest**, **SVC**, and **K-Nearest Neighbors (KNN)**.
- **🗺️ Decision Boundary Visualizer:** 2D contour visualization showing how models separate placed vs. non-placed students.
- **🎯 Live Placement Evaluator:** Interactive sliders for candidate IQ and CGPA with real-time probability gauge and strategic feedback.
- **📁 Batch Prediction:** Upload custom CSV files with candidate metrics to generate and download batch placement predictions.

---

## 📁 Repository Structure

```
.
├── app.py                   # Streamlit Web Application
├── Untitled.ipynb           # Machine Learning Analysis Notebook
├── placement_dataset.csv    # Student Placement Dataset
├── profile_photo.jpg        # Profile Photo for Dashboard Sidebar
├── requirements.txt         # Python Package Dependencies
└── README.md                # Project Documentation
```

---

## 📊 Dataset Features

| Feature Column | Type | Description |
| :--- | :--- | :--- |
| `iq` | Integer / Float | Candidate Intelligence Quotient Score (50 - 180) |
| `cgpa` | Float | Cumulative Grade Point Average (0.0 - 10.0) |
| `placement` | Binary (0 / 1) | Target Label: `1` = Placed, `0` = Not Placed |

---

## 🚀 Local Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/RajSingh2006-git/Student_Placement-Part_1.git
   cd Student_Placement-Part_1
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```

4. **Access the application in your browser:**
   ```
   http://localhost:8501
   ```

---

## 🌐 Cloud Deployment (Streamlit Community Cloud)

You can easily deploy this repository to **Streamlit Community Cloud**:

1. Push this repository to your GitHub account (`RajSingh2006-git/Student_Placement-Part_1`).
2. Visit [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New app"** and select:
   - **Repository:** `RajSingh2006-git/Student_Placement-Part_1`
   - **Branch:** `main`
   - **Main file path:** `app.py`
4. Click **Deploy!**

---

## 🛠️ Built With

- **Python** - Core Programming Language
- **Streamlit** - Web Application Framework
- **Scikit-Learn** - Machine Learning Library
- **Pandas & NumPy** - Data Manipulation
- **Plotly, Matplotlib & Seaborn** - Data Visualization

---

## 👤 Author

- **Raj Singh** - [RajSingh2006-git](https://github.com/RajSingh2006-git)
>>>>>>> b07a725 (Initial commit: Student Placement ML dataset, Streamlit app, notebook, and documentation)
