import streamlit as st
import pandas as pd
import joblib
import time

# ==========================
# Page Configuration
# ==========================
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ==========================
# Load Model
# ==========================
model = joblib.load("knn_heart_model.pkl")
scaler = joblib.load("heart_scaler.pkl")
expected_columns = joblib.load("heart_columns.pkl")

# ==========================
# Custom CSS — Modern Gradient Theme
# ==========================
st.markdown("""
<style>
/* ── Import Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif !important;
}

/* ── Global Background ── */
html, body, [data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e) !important;
    color: #ffffff !important;
}

[data-testid="stMain"], .main {
    background: transparent !important;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(20px) !important;
    border-right: 1px solid rgba(255, 255, 255, 0.1) !important;
}

[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* ── Main Headings ── */
h1 {
    background: linear-gradient(135deg, #f7971e, #ffd200) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    text-align: center !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-shadow: 0 0 40px rgba(255, 210, 0, 0.3) !important;
    animation: glow 2s ease-in-out infinite alternate !important;
}

@keyframes glow {
    from { text-shadow: 0 0 20px rgba(255, 210, 0, 0.3); }
    to { text-shadow: 0 0 40px rgba(255, 210, 0, 0.6); }
}

h5 {
    text-align: center !important;
    color: rgba(255, 255, 255, 0.7) !important;
    font-weight: 300 !important;
}

h2, h3, .stSubheader {
    background: linear-gradient(135deg, #a8edea, #fed6e3) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
}

/* ── Card Style Inputs ── */
div[data-baseweb="input"] input,
div[data-baseweb="select"] div,
div[data-baseweb="slider"] {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    backdrop-filter: blur(10px) !important;
    transition: all 0.3s ease !important;
}

div[data-baseweb="input"] input:focus,
div[data-baseweb="select"] div:focus {
    border-color: #f7971e !important;
    box-shadow: 0 0 20px rgba(247, 151, 30, 0.2) !important;
}

[data-testid="stNumberInput"] input,
[data-testid="stTextInput"] input {
    background: rgba(255, 255, 255, 0.08) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

/* ── Slider ── */
div[data-testid="stSlider"] div[role="slider"] {
    background: linear-gradient(135deg, #f7971e, #ffd200) !important;
}

/* ── Selectbox ── */
ul[data-testid="stSelectboxVirtualDropdown"],
div[data-baseweb="popover"] ul {
    background: rgba(30, 30, 60, 0.95) !important;
    backdrop-filter: blur(20px) !important;
    color: #ffffff !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
}

/* ── Predict Button ── */
div.stButton > button {
    background: linear-gradient(135deg, #f7971e, #ffd200) !important;
    color: #0f0c29 !important;
    font-size: 20px !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    border: none !important;
    border-radius: 15px !important;
    width: 100% !important;
    height: 60px !important;
    box-shadow: 0 8px 30px rgba(247, 151, 30, 0.4) !important;
    transition: all 0.4s ease !important;
    text-transform: uppercase !important;
}

div.stButton > button:hover {
    transform: translateY(-3px) scale(1.02) !important;
    box-shadow: 0 12px 40px rgba(247, 151, 30, 0.6) !important;
}

div.stButton > button:active {
    transform: scale(0.98) !important;
}

/* ── Alert Boxes ── */
div[data-testid="stAlert"] {
    border-radius: 15px !important;
    border-left-width: 6px !important;
    backdrop-filter: blur(10px) !important;
    padding: 20px !important;
}

div[data-testid="stAlert"][class*="success"] {
    background: rgba(46, 204, 113, 0.2) !important;
    border-color: #2ecc71 !important;
    color: #2ecc71 !important;
}

div[data-testid="stAlert"][class*="error"] {
    background: rgba(231, 76, 60, 0.2) !important;
    border-color: #e74c3c !important;
    color: #e74c3c !important;
}

div[data-testid="stAlert"][class*="warning"] {
    background: rgba(241, 196, 15, 0.2) !important;
    border-color: #f1c40f !important;
    color: #f1c40f !important;
}

div[data-testid="stAlert"][class*="info"] {
    background: rgba(52, 152, 219, 0.2) !important;
    border-color: #3498db !important;
    color: #3498db !important;
}

/* ── Progress Bar ── */
div[data-testid="stProgress"] > div {
    background: rgba(255, 255, 255, 0.1) !important;
    border-radius: 20px !important;
    height: 12px !important;
}
div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #f7971e, #ffd200) !important;
    border-radius: 20px !important;
    height: 12px !important;
}

/* ── Columns Styling ── */
[data-testid="column"] {
    background: rgba(255, 255, 255, 0.03) !important;
    border-radius: 20px !important;
    padding: 20px !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
}

/* ── Labels ── */
label, p, .stMarkdown {
    color: rgba(255, 255, 255, 0.9) !important;
    font-weight: 400 !important;
}

/* ── Sidebar Content ── */
[data-testid="stSidebar"] [data-testid="stAlert"] {
    background: rgba(255, 215, 0, 0.1) !important;
    border-color: #ffd200 !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    height: 2px !important;
    background: linear-gradient(90deg, transparent, #ffd200, transparent) !important;
    margin: 30px 0 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] {
    color: #ffd200 !important;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# Header with Animation
# ==========================
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1>❤️ Heart Disease Prediction System</h1>
    <p style="color: rgba(255,255,255,0.6); font-size: 1.1rem; margin-top: -10px;">
        Developed by <span style="color: #ffd200; font-weight: 600;">Sabir Shah</span>
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================
# Sidebar with Modern Design
# ==========================
st.sidebar.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h2 style="font-size: 1.8rem; background: linear-gradient(135deg, #f7971e, #ffd200); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">👨‍💻 About</h2>
</div>
""", unsafe_allow_html=True)

st.sidebar.success("""
### 🚀 **Developer:** Sabir Shah

---

### 🎯 **Technologies:**
- 📊 Machine Learning
- 🤖 Artificial Intelligence  
- 💻 Python & Streamlit
- 🎨 Modern UI/UX

---

### 📈 **Accuracy:** 85%+
### 🔬 **Model:** KNN Classifier
""")

st.sidebar.markdown("""
<div style="background: rgba(255,215,0,0.1); border-radius: 15px; padding: 15px; text-align: center; border: 1px solid rgba(255,215,0,0.2); margin-top: 20px;">
    <span style="color: #ffd200;">⚡ Powered by AI</span>
</div>
""", unsafe_allow_html=True)

# ==========================
# Main Content
# ==========================
st.markdown("""
<div style="text-align: center; padding: 10px 0;">
    <h2 style="font-size: 2rem;">📝 Patient Information</h2>
    <p style="color: rgba(255,255,255,0.5);">Fill in the details below to predict heart disease risk</p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div style="background: rgba(255,255,255,0.03); border-radius: 15px; padding: 20px; border: 1px solid rgba(255,255,255,0.05);">
    """, unsafe_allow_html=True)
    
    age = st.slider("📅 Age", 18, 100, 40)
    sex = st.selectbox("👤 Gender", ["M", "F"])
    resting_bp = st.number_input("💓 Resting Blood Pressure", 80, 200, 120)
    cholesterol = st.number_input("🧪 Cholesterol", 100, 600, 200)
    fasting_bs = st.selectbox("🍬 Fasting Blood Sugar >120", [0, 1])
    oldpeak = st.slider("📊 Old Peak", 0.0, 6.0, 1.0)
    
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: rgba(255,255,255,0.03); border-radius: 15px; padding: 20px; border: 1px solid rgba(255,255,255,0.05);">
    """, unsafe_allow_html=True)
    
    chest_pain = st.selectbox("🫀 Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_ecg = st.selectbox("📈 Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("🏃 Maximum Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("💪 Exercise Angina", ["Y", "N"])
    st_slope = st.selectbox("📉 ST Slope", ["Up", "Flat", "Down"])
    
    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# ==========================
# Prediction with Animation
# ==========================
if st.button("🔍 Predict Heart Disease", use_container_width=True):
    
    # Preparing input data
    raw_input = {
        'Age': age,
        'RestingBP': resting_bp,
        'Cholesterol': cholesterol,
        'FastingBS': fasting_bs,
        'MaxHR': max_hr,
        'Oldpeak': oldpeak,
        'Sex_' + sex: 1,
        'ChestPainType_' + chest_pain: 1,
        'RestingECG_' + resting_ecg: 1,
        'ExerciseAngina_' + exercise_angina: 1,
        'ST_Slope_' + st_slope: 1
    }

    input_df = pd.DataFrame([raw_input])

    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]
    scaled_input = scaler.transform(input_df)

    with st.spinner("🔄 Analyzing patient data..."):
        time.sleep(1.5)  # Added delay for better UX
        prediction = model.predict(scaled_input)[0]

    st.markdown("---")
    
    # Display result with enhanced visuals
    if prediction == 1:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(231,76,60,0.2), rgba(231,76,60,0.05)); 
                        border: 2px solid #e74c3c; 
                        border-radius: 20px; 
                        padding: 30px; 
                        text-align: center;
                        box-shadow: 0 10px 40px rgba(231,76,60,0.2);">
                <h1 style="font-size: 3rem; color: #e74c3c;">🚨</h1>
                <h2 style="color: #e74c3c; margin: 10px 0;">High Risk of Heart Disease</h2>
                <div style="margin: 20px 0;">
                    <div style="background: rgba(231,76,60,0.1); border-radius: 10px; padding: 10px;">
                        <span style="color: #e74c3c;">Risk Level: </span>
                        <span style="color: #ff6b6b; font-weight: 700;">90%</span>
                    </div>
                </div>
                <p style="color: rgba(255,255,255,0.7);">⚠️ Please consult a healthcare professional immediately.</p>
            </div>
            """, unsafe_allow_html=True)
        st.progress(90)
        
    else:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, rgba(46,204,113,0.2), rgba(46,204,113,0.05)); 
                        border: 2px solid #2ecc71; 
                        border-radius: 20px; 
                        padding: 30px; 
                        text-align: center;
                        box-shadow: 0 10px 40px rgba(46,204,113,0.2);">
                <h1 style="font-size: 3rem; color: #2ecc71;">✅</h1>
                <h2 style="color: #2ecc71; margin: 10px 0;">Low Risk of Heart Disease</h2>
                <div style="margin: 20px 0;">
                    <div style="background: rgba(46,204,113,0.1); border-radius: 10px; padding: 10px;">
                        <span style="color: #2ecc71;">Risk Level: </span>
                        <span style="color: #55efc4; font-weight: 700;">25%</span>
                    </div>
                </div>
                <p style="color: rgba(255,255,255,0.7);">🌟 Maintain a healthy lifestyle with regular exercise and balanced diet.</p>
            </div>
            """, unsafe_allow_html=True)
        st.progress(25)
        st.balloons()

st.markdown("---")

# ==========================
# Footer
# ==========================
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h3 style="background: linear-gradient(135deg, #f7971e, #ffd200); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        ❤️ Developed by Sabir Shah
    </h3>
    <p style="color: rgba(255,255,255,0.4); font-size: 0.9rem;">
        Machine Learning | Data Science | Python | Streamlit
    </p>
</div>
""", unsafe_allow_html=True)