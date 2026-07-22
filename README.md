# 🎓 Student Placement Analytics & Machine Learning Predictor

An interactive Machine Learning web application built with **Streamlit**, **Scikit-Learn**, and **Plotly** to analyze student academic metrics (IQ & CGPA) and predict campus placement status.

[![Live Demo](https://img.shields.io/badge/Streamlit_App-Live_Demo-FF4B4B?style=for-the-badge&logo=streamlit)](https://studentplacement-part1-8d9iurtpukngtxj6z86cfx.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange?style=for-the-badge&logo=scikitlearn)

---

## 🌐 Live Web Application

The app is live and deployed on Streamlit Community Cloud:
👉 **[Click Here to Access Live App](https://studentplacement-part1-8d9iurtpukngtxj6z86cfx.streamlit.app/)**

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

## 🌐 Deployment Details

- **Deployment URL:** [https://studentplacement-part1-8d9iurtpukngtxj6z86cfx.streamlit.app/](https://studentplacement-part1-8d9iurtpukngtxj6z86cfx.streamlit.app/)
- **Hosting Platform:** Streamlit Community Cloud
- **Repository:** `RajSingh2006-git/Student_Placement-Part_1`
- **Main Entry Point:** `app.py`

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
