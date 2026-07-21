import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Student Placement Analytics & Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Main Background & Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Container */
    .header-container {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 40%, #4338ca 100%);
        padding: 2.2rem 2.5rem;
        border-radius: 20px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(67, 56, 202, 0.3);
    }
    .header-title {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .header-subtitle {
        font-size: 1.1rem;
        color: #c7d2fe;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 16px;
        border: 1px solid #e0e7ff;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.06);
    }
    .metric-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        color: #1e1b4b;
        margin-top: 0.4rem;
    }
    .metric-sub {
        font-size: 0.85rem;
        font-weight: 500;
        margin-top: 0.2rem;
    }

    /* Prediction Outcome Cards */
    .result-card-placed {
        background: linear-gradient(135deg, #ecfdf5 0%, #d1fae5 100%);
        border: 2px solid #10b981;
        padding: 2rem;
        border-radius: 20px;
        color: #065f46;
        text-align: center;
        margin-top: 1rem;
    }
    .result-card-not-placed {
        background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
        border: 2px solid #f43f5e;
        padding: 2rem;
        border-radius: 20px;
        color: #9f1239;
        text-align: center;
        margin-top: 1rem;
    }

    /* Tab & Container Tweaks */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        padding: 10px 20px;
        background-color: #f1f5f9;
        border: none;
        font-weight: 600;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4338ca !important;
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loader
# ---------------------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_dir, 'placement_dataset.csv')
    if not os.path.exists(file_path):
        file_path = 'placement_dataset.csv'
    
    df = pd.read_csv(file_path)
    # Cleaning as in notebook
    duplicates_count = df.duplicated().sum()
    df_clean = df.drop_duplicates().reset_index(drop=True)
    return df, df_clean, duplicates_count

df_raw, df, duplicate_rows = load_data()

# ---------------------------------------------------------
# Sidebar Profile Photo & Navigation
# ---------------------------------------------------------
import base64
profile_photo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'profile_photo.jpg')

if os.path.exists(profile_photo_path):
    with open(profile_photo_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode()
    st.sidebar.markdown(
        f"""
        <div style="text-align: center; margin-bottom: 1.2rem;">
            <img src="data:image/jpeg;base64,{img_b64}" 
                 style="width: 130px; height: 130px; border-radius: 50%; object-fit: cover; border: 4px solid #4338ca; box-shadow: 0 6px 15px rgba(67, 56, 202, 0.25);">
        </div>
        """, 
        unsafe_allow_html=True
    )
else:
    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)

st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Overview & EDA",
        "📊 Data Visualization",
        "⚙️ Machine Learning Models",
        "🎯 Placement Predictor",
        "📁 Batch Prediction"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🛠 Dataset Info")
st.sidebar.info(
    f"**Total Records:** {len(df_raw)}\n\n"
    f"**Cleaned Records:** {len(df)}\n\n"
    f"**Duplicates Removed:** {duplicate_rows}\n\n"
    f"**Features:** IQ, CGPA"
)

# ---------------------------------------------------------
# Header Banner
# ---------------------------------------------------------
st.markdown("""
<div class="header-container">
    <div class="header-title">🎓 Student Placement Intelligence</div>
    <div class="header-subtitle">Interactive ML Analytics, Decision Boundary Visualizations & Placement Prediction System</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Module 1: Overview & EDA
# ---------------------------------------------------------
if page == "🏠 Overview & EDA":
    st.subheader("📌 Executive Summary & Key Metrics")
    
    placed_df = df[df['placement'] == 1]
    unplaced_df = df[df['placement'] == 0]
    total_students = len(df)
    placed_count = len(placed_df)
    unplaced_count = len(unplaced_df)
    placement_rate = (placed_count / total_students) * 100
    
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Total Students</div>
            <div class="metric-value">{total_students}</div>
            <div class="metric-sub" style="color: #64748b;">Cleaned Records</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Placed</div>
            <div class="metric-value" style="color: #10b981;">{placed_count}</div>
            <div class="metric-sub" style="color: #10b981;">{placement_rate:.1f}% Success</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Not Placed</div>
            <div class="metric-value" style="color: #f43f5e;">{unplaced_count}</div>
            <div class="metric-sub" style="color: #f43f5e;">{100-placement_rate:.1f}% Pending</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg. CGPA</div>
            <div class="metric-value">{df['cgpa'].mean():.2f}</div>
            <div class="metric-sub" style="color: #64748b;">Scale 0.0 - 10.0</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Avg. IQ</div>
            <div class="metric-value">{df['iq'].mean():.1f}</div>
            <div class="metric-sub" style="color: #64748b;">Intelligence Quotient</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    col_left, col_right = st.columns([1.2, 1])
    
    with col_left:
        st.subheader("📋 Dataset Preview")
        st.dataframe(
            df.style.highlight_max(subset=['cgpa', 'iq'], color='#dbeafe')
                    .highlight_between(left=1, right=1, subset=['placement'], color='#d1fae5'),
            height=320,
            use_container_width=True
        )
        
    with col_right:
        st.subheader("📊 Summary Statistics")
        desc = df.describe().T[['mean', 'std', 'min', '50%', 'max']]
        desc.columns = ['Mean', 'Std Dev', 'Min', 'Median', 'Max']
        st.dataframe(desc.style.format("{:.2f}"), use_container_width=True)
        
        st.markdown("##### 🔍 Data Quality Check")
        st.success(f"✅ Null values count: {df.isnull().sum().sum()}")
        st.info(f"ℹ️ Original shape: {df_raw.shape} | Cleaned shape: {df.shape}")

# ---------------------------------------------------------
# Module 2: Data Visualization
# ---------------------------------------------------------
elif page == "📊 Data Visualization":
    st.subheader("🎨 Exploratory Visual Analytics")
    
    t1, t2, t3 = st.tabs(["🎯 Placement Scatter Plot", "📈 Feature Distributions", "🔥 Correlation Matrix"])
    
    with t1:
        st.markdown("#### IQ vs CGPA by Placement Status")
        fig_scatter = px.scatter(
            df,
            x='cgpa',
            y='iq',
            color=df['placement'].astype(str).map({'0': 'Not Placed', '1': 'Placed'}),
            color_discrete_map={'Placed': '#10b981', 'Not Placed': '#f43f5e'},
            hover_data=['cgpa', 'iq', 'placement'],
            labels={'cgpa': 'CGPA (0 - 10)', 'iq': 'IQ Score', 'color': 'Placement Status'},
            title="Candidate Placement Distribution",
            template="plotly_white",
            height=500
        )
        fig_scatter.update_traces(marker=dict(size=10, opacity=0.8, line=dict(width=1, color='White')))
        fig_scatter.update_layout(font=dict(family="Inter"))
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    with t2:
        c_dist1, c_dist2 = st.columns(2)
        with c_dist1:
            st.markdown("##### CGPA Distribution")
            fig_cgpa = px.histogram(
                df,
                x='cgpa',
                color=df['placement'].astype(str).map({'0': 'Not Placed', '1': 'Placed'}),
                barmode="overlay",
                color_discrete_map={'Placed': '#10b981', 'Not Placed': '#f43f5e'},
                marginal="box",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_cgpa, use_container_width=True)
            
        with c_dist2:
            st.markdown("##### IQ Distribution")
            fig_iq = px.histogram(
                df,
                x='iq',
                color=df['placement'].astype(str).map({'0': 'Not Placed', '1': 'Placed'}),
                barmode="overlay",
                color_discrete_map={'Placed': '#10b981', 'Not Placed': '#f43f5e'},
                marginal="box",
                template="plotly_white",
                height=400
            )
            st.plotly_chart(fig_iq, use_container_width=True)
            
    with t3:
        st.markdown("#### Correlation Analysis")
        corr = df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".3f",
            color_continuous_scale="Viridis",
            title="Feature Correlation Matrix",
            template="plotly_white",
            aspect="auto"
        )
        st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------------
# Module 3: Machine Learning Models & Decision Boundary
# ---------------------------------------------------------
elif page == "⚙️ Machine Learning Models":
    st.subheader("🤖 Model Training, Evaluation & Decision Boundary")
    
    st.sidebar.markdown("### 🎛 Model Configuration")
    test_size = st.sidebar.slider("Test Split Size (%)", 5, 40, 10, step=5) / 100.0
    random_state = st.sidebar.number_input("Random State", value=42, step=1)
    
    model_choice = st.sidebar.selectbox(
        "Select Algorithm",
        [
            "Logistic Regression (Notebook Model)",
            "Random Forest Classifier",
            "Support Vector Classifier (SVC)",
            "K-Nearest Neighbors (KNN)"
        ]
    )
    
    # Train-test split
    X = df[['iq', 'cgpa']]
    y = df['placement']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Model Selection logic
    if "Logistic Regression" in model_choice:
        model = LogisticRegression()
    elif model_choice == "Random Forest Classifier":
        n_estimators = st.sidebar.slider("N Estimators", 10, 200, 100)
        model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    elif model_choice == "Support Vector Classifier (SVC)":
        C_val = st.sidebar.slider("C (Regularization)", 0.1, 10.0, 1.0)
        model = SVC(C=C_val, probability=True, random_state=random_state)
    elif model_choice == "K-Nearest Neighbors (KNN)":
        k_val = st.sidebar.slider("K Neighbors", 1, 15, 5)
        model = KNeighborsClassifier(n_neighbors=k_val)
        
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc * 100:.2f}%")
    m2.metric("Precision", f"{prec * 100:.2f}%")
    m3.metric("Recall", f"{rec * 100:.2f}%")
    m4.metric("F1 Score", f"{f1 * 100:.2f}%")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    c_left, c_right = st.columns([1.1, 1])
    
    with c_left:
        st.markdown("#### 🗺 Decision Boundary Visualizer")
        
        # Grid generation for decision boundary
        x_min, x_max = X['iq'].min() - 5, X['iq'].max() + 5
        y_min, y_max = X['cgpa'].min() - 0.5, X['cgpa'].max() + 0.5
        
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200)
        )
        
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        grid_scaled = scaler.transform(grid_points)
        Z = model.predict(grid_scaled)
        Z = Z.reshape(xx.shape)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
        
        scatter = ax.scatter(
            df['iq'], df['cgpa'],
            c=df['placement'],
            cmap=plt.cm.coolwarm,
            edgecolors='k',
            linewidths=0.7,
            s=50
        )
        ax.set_xlabel("IQ Score", fontsize=11, fontweight='bold')
        ax.set_ylabel("CGPA (0 - 10)", fontsize=11, fontweight='bold')
        ax.set_title(f"Decision Boundary ({model_choice})", fontsize=13, fontweight='bold')
        ax.grid(True, linestyle='--', alpha=0.5)
        
        st.pyplot(fig)
        
    with c_right:
        st.markdown("#### 📋 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(
            cm,
            text_auto=True,
            x=['Not Placed (0)', 'Placed (1)'],
            y=['Not Placed (0)', 'Placed (1)'],
            color_continuous_scale="Blues",
            labels=dict(x="Predicted", y="Actual"),
            template="plotly_white"
        )
        st.plotly_chart(fig_cm, use_container_width=True)
        
        st.markdown("#### 📄 Classification Report")
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).T
        st.dataframe(report_df.style.format("{:.3f}"), use_container_width=True)

# ---------------------------------------------------------
# Module 4: Placement Predictor
# ---------------------------------------------------------
elif page == "🎯 Placement Predictor":
    st.subheader("⚡ Live Placement Chance Evaluator")
    st.markdown("Adjust the candidate's **IQ Score** and **CGPA** below to evaluate their placement probability in real-time.")
    
    # Pre-train default LogisticRegression model on all cleaned data for best prediction stability
    X = df[['iq', 'cgpa']]
    y = df['placement']
    
    scaler_full = StandardScaler()
    X_scaled = scaler_full.fit_transform(X)
    
    model_full = LogisticRegression()
    model_full.fit(X_scaled, y)
    
    c_in1, c_in2 = st.columns(2)
    
    with c_in1:
        student_iq = st.slider(
            "🧠 Candidate IQ Score",
            min_value=50,
            max_value=180,
            value=105,
            step=1,
            help="Select student intelligence quotient score."
        )
        
    with c_in2:
        student_cgpa = st.slider(
            "🎓 Candidate CGPA",
            min_value=0.0,
            max_value=10.0,
            value=7.50,
            step=0.05,
            help="Select student cumulative grade point average."
        )
        
    # Scale input
    input_data = np.array([[student_iq, student_cgpa]])
    input_scaled = scaler_full.transform(input_data)
    
    pred_class = model_full.predict(input_scaled)[0]
    pred_probs = model_full.predict_proba(input_scaled)[0]
    placed_prob = pred_probs[1] * 100
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    res_col1, res_col2 = st.columns([1, 1.2])
    
    with res_col1:
        if pred_class == 1:
            st.markdown(f"""
            <div class="result-card-placed">
                <h2 style="margin:0; font-size:2.2rem;">🎉 Placed!</h2>
                <p style="font-size:1.2rem; margin-top:0.5rem; font-weight:600;">High Chance of Campus Placement</p>
                <div style="font-size:3rem; font-weight:800; margin-top:1rem;">{placed_prob:.1f}%</div>
                <div style="font-size:0.9rem; color:#047857;">Estimated Placement Probability</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-card-not-placed">
                <h2 style="margin:0; font-size:2.2rem;">⚠️ Not Placed</h2>
                <p style="font-size:1.2rem; margin-top:0.5rem; font-weight:600;">Requires Academic & Skill Improvement</p>
                <div style="font-size:3rem; font-weight:800; margin-top:1rem;">{placed_prob:.1f}%</div>
                <div style="font-size:0.9rem; color:#b91c1c;">Estimated Placement Probability</div>
            </div>
            """, unsafe_allow_html=True)
            
    with res_col2:
        st.markdown("#### ⏱ Probability Gauge")
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = placed_prob,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Placement Likelihood (%)", 'font': {'size': 18}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#4338ca"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 45], 'color': '#ffe4e6'},
                    {'range': [45, 70], 'color': '#fef3c7'},
                    {'range': [70, 100], 'color': '#d1fae5'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)
        
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("#### 💡 Candidate Recommendation & Strategy")
    if student_cgpa < 7.0:
        st.warning("👉 **Key Takeaway:** CGPA is below the typical threshold (7.0). Focusing on improving academic grades will significantly boost placement odds.")
    elif student_iq < 95:
        st.info("👉 **Key Takeaway:** CGPA is good! Strengthening problem-solving, logical reasoning, and aptitude test practice will solidify placement readiness.")
    else:
        st.success("👉 **Key Takeaway:** Excellent academic profile and strong aptitude metrics! The candidate is in a prime position for top company recruitment drives.")

# ---------------------------------------------------------
# Module 5: Batch Prediction
# ---------------------------------------------------------
elif page == "📁 Batch Prediction":
    st.subheader("📁 Batch Placement Prediction via CSV")
    st.markdown("Upload a custom CSV file containing `iq` and `cgpa` columns to run batch predictions.")
    
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    
    # Pre-train baseline model
    X = df[['iq', 'cgpa']]
    y = df['placement']
    scaler_batch = StandardScaler()
    X_scaled = scaler_batch.fit_transform(X)
    model_batch = LogisticRegression()
    model_batch.fit(X_scaled, y)
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.write("##### 📄 Uploaded Data Preview:")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            if 'iq' in batch_df.columns and 'cgpa' in batch_df.columns:
                batch_scaled = scaler_batch.transform(batch_df[['iq', 'cgpa']])
                preds = model_batch.predict(batch_scaled)
                probs = model_batch.predict_proba(batch_scaled)[:, 1] * 100
                
                batch_df['Predicted_Placement'] = preds
                batch_df['Placement_Probability_%'] = np.round(probs, 2)
                batch_df['Status'] = batch_df['Predicted_Placement'].map({1: 'Placed', 0: 'Not Placed'})
                
                st.markdown("---")
                st.markdown("##### 🚀 Prediction Results:")
                st.dataframe(batch_df, use_container_width=True)
                
                csv_data = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Predictions CSV",
                    data=csv_data,
                    file_name="placement_predictions_output.csv",
                    mime="text/csv"
                )
            else:
                st.error("❌ CSV must contain both 'iq' and 'cgpa' columns.")
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("💡 **Sample Format:** CSV should contain columns named `iq` and `cgpa`.")
        sample_data = pd.DataFrame({
            'iq': [105, 95, 120, 88, 110],
            'cgpa': [7.5, 6.2, 8.8, 5.5, 8.1]
        })
        st.dataframe(sample_data, use_container_width=True)
        csv_sample = sample_data.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="⬇️ Download Sample Input CSV",
            data=csv_sample,
            file_name="sample_placement_input.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem; padding: 1rem;'>"
    "Student Placement Machine Learning Application • Developed with Streamlit, Scikit-Learn & Plotly"
    "</div>",
    unsafe_allow_html=True
)
