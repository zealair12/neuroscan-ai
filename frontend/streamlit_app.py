import streamlit as st
import requests
from PIL import Image
import io

API_URL = "http://127.0.0.1:5000/predict"

# -----------------------
# APP CONFIG
# -----------------------
st.set_page_config(
    page_title="🧠 NeuroScan AI",
    page_icon="🧠",
    layout="centered"
)

# Custom CSS for style upgrades
st.markdown("""
    <style>
        .main {
            background-color: #f5f7fb;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border-radius: 8px;
            padding: 0.6em 1.2em;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #388e3c;
        }
        .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
            color: #004080;
        }
    </style>
""", unsafe_allow_html=True)

# -----------------------
# SIDEBAR NAV / INFO
# -----------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2942/2942877.png", width=100)
st.sidebar.title("🔎 How It Works")
st.sidebar.markdown("""
Upload an MRI scan to get a rapid AI-based classification.

**Supported formats**: JPG, PNG  
**Model**: CNN (pretrained)  
**Confidence**: Returns a % likelihood  
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 For internal diagnostic use by medical professionals only.")

# -----------------------
# MAIN APP AREA
# -----------------------
st.markdown("## 🧠 NeuroScan AI")
st.markdown("### Precision MRI Tumor Detection")
st.markdown("Upload a brain scan image. The AI model will classify it in real time.")

uploaded_file = st.file_uploader("📤 Upload an MRI Scan", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="🖼️ Uploaded MRI", use_container_width=True)

    if st.button("🔍 Analyze Scan"):
        with st.spinner("🧬 Running diagnostic model..."):
            try:
                response = requests.post(
                    API_URL,
                    files={"image": uploaded_file.getvalue()}
                )

                if response.status_code == 200:
                    result = response.json()
                    prediction = result['prediction']
                    confidence = result['confidence']

                    st.success(f"🧾 **Prediction**: `{prediction}`")
                    st.info(f"📈 **Confidence**: `{confidence:.2%}`")
                else:
                    st.error("❌ Server returned an error.")

            except Exception as e:
                st.error(f"🚨 Couldn't connect to the backend: {e}")

else:
    st.warning("🧑‍⚕️ Please upload an image before running analysis.")
