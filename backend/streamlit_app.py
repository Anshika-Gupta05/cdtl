
import streamlit as st
from PIL import Image
import torch
from utils.model_utils import load_model, predict_image, apply_gradcam
import os
import sys
import base64

# Skip torch internal C++ classes during hot-reloading
sys.modules["torch.classes"] = None


image_url = "https://ik.imagekit.io/ag1qvulim/lungs.png?updatedAt=1748351829845"
st.set_page_config(
    page_title="RespiraScan",
    page_icon=image_url,  # This will only work in some environments, fallback to emoji otherwise
    layout="wide",
    initial_sidebar_state="expanded"
)

# === INJECT CSS ===
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=EB+Garamond&display=swap');

    :root {
        --primary-color: #A8D6EB;
        --secondary-color: #1B3A57;
        --tertiary-color: #F5F9FC;
        --quarter-color: #0B3649;
        --accent-color: #C9E1EC;
        --main-font: 'EB Garamond', serif;
    }

    html, body, [class*="css"]  {
        font-family: var(--main-font) !important;
    }

    .stApp {
        font-family: var(--main-font) !important;

    }

    h1, h2, h3, h4 {
        font-weight: 800 !important;
                    font-family: var(--main-font) !important;

    }

    span{
            font-family: var(--main-font) !important;
            font-weight: 600 !important;
            color: white !important;
    }
    label,p,div{
        font-family: var(--main-font) !important;

            }
    .custom-hr {
        border: none;
        height: 2px;
        background-color: var(--secondary-color) !important;
        margin: 1.5rem 0;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(to right, #0B3C51, #0B3C51) !important;
        color: white !important;
        border-right: 1px solid #ccc;
        padding: 1rem;
                    font-family: var(--main-font) !important;

    }
    section[data-testid="stSidebar"] * {
    color: white !important;
}

    button[kind="primary"] {
        background-color: var(--primary-color) !important;
        color: white !important;
        font-weight: 700;
        border-radius: 8px;
        border: none;
        padding: 0.5rem 1rem;
        transition: background-color 0.3s ease;
                    font-family: var(--main-font) !important;

    }

    button[kind="primary"]:hover {
        background-color: var(--accent-color) !important;
        color: white !important;
                    font-family: var(--main-font) !important;

    }

    .stFileUploader label {
        font-weight: 600;
        font-family: var(--main-font) !important;

    }

    .stSpinner > div {
        animation: spin 2s linear infinite;
    }

    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    div[data-testid="stMetricValue"] {
        font-weight: 800;
        font-size: 22px;
        font-family: var(--main-font) !important;

    }

    .stTabs [data-baseweb="tab-list"] {
        background-color: var(--tertiary-color) !important;
        color: white !important;
        border-radius: 10px;
        margin-bottom: 1rem;
                    font-family: var(--main-font) !important;

    }

    .stTabs [data-baseweb="tab"] {
        color: white !important;
                    font-family: var(--main-font) !important;

    }

    .stImage > figcaption {
        font-weight: 600;
        margin-top: 0.3rem;
                    font-family: var(--main-font) !important;

    }

    footer, .css-qri22k {
                    font-family: var(--main-font) !important;

    }
            
    </style>
""", unsafe_allow_html=True)

# === SIDEBAR ===
st.sidebar.markdown(
    "<h2 style='color: white; margin-bottom: 0.5rem; font-weight: 800;'>🧭 Navigation</h2>",
    unsafe_allow_html=True
)

page = st.sidebar.selectbox(
    "",
    ["Home", "About"],
    label_visibility="collapsed"
)

st.sidebar.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

st.sidebar.markdown(
    f"""
    <div style='display: flex; align-items: center; margin-bottom: 0.1rem;'>
        <img src="{image_url}" 
             alt="Logo" 
             style="height: 30px; width: 30px; margin-right: 10px;">
        <span style="font-size: 20px; font-weight: bold; font-family: "EB Garamond", serif;">
            RespiraScan
        </span>
    </div>
    <div style='color: white; font-size: 14px; line-height: 1.5; text-align: left;'>
        <p>AI-powered diagnostic assistant for detecting lung diseases from chest X-rays<br><br>
        © 2025 RespiraScan — Developed with ❤️ using Streamlit, PyTorch, and Grad-CAM<br>
       </p>  
    </div>
    """,
    unsafe_allow_html=True,
)

# === LOAD MODEL WITH CACHING ===
@st.cache_resource(show_spinner=False)
def load_cached_model():
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    model_path = os.path.join(BASE_DIR, 'models', 'final_lung_disease_model.pth')
    return load_model(model_path)

model = load_cached_model()

# === HOME PAGE ===
if page == "Home":
    
   # Display the header with embedded base64 image
    st.markdown(
    f"""
    <div style='display: flex; justify-content: center; align-items: center; margin-bottom: 0.1rem;'>
        <img src="{image_url}" 
             alt="Logo" 
             style="height: 50px; width: 50px; margin-right: 10px;">
        <span style="font-size: 40px; font-weight: bold; font-family: 'EB Garamond', serif;">
            RespiraScan
        </span>
    </div>
    """,
    unsafe_allow_html=True
)

    st.markdown("<p style='text-align: center; font-size:20px; font-weight:600; margin-top:0; margin-bottom:0.3rem;'>Early Detection, Healthier Lungs</p>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align: center; font-size:16px; font-style: italic; margin-top:0; margin-bottom:1rem;'>"
        "Detect Pneumonia, Tuberculosis, and Pulmonary Fibrosis effortlessly using AI-driven analysis of Chest X-rays."
        "</p>", unsafe_allow_html=True)
    
    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)
    

    uploaded_file = st.file_uploader(
        "Upload a chest X-ray image:",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible"
    )
    


    if uploaded_file is not None:
        
        try:
            image = Image.open(uploaded_file).convert('RGB')

            # Use a more balanced layout: 1:1 or 4:3 for image vs results
            col1, col2 = st.columns([4, 5])

            with col1:
                st.subheader("Input Image")
                st.image(image, use_container_width=True, clamp=True)
            
            with st.spinner(""):
                predictions = predict_image(image, model)
                gradcam_maps = apply_gradcam(image.copy(), model)

            with col2:
                st.subheader("Diagnosis Results")
                max_disease = max(
                    ((disease, info) for disease, info in predictions.items()
                     if info['label'] == 'Disease'),
                    key=lambda x: x[1]['confidence'],
                    default=(None, None)
                )

                if max_disease[0]:
                    st.warning(
                        f"🚨 Highest Confidence: {max_disease[0]} "
                        f"({max_disease[1]['confidence']*100:.1f}%)"
                    )
                else:
                    st.success("✅ No abnormalities detected")

                for disease, result in predictions.items():
                    emoji = "⚠️" if result['label'] == 'Disease' else "✅"
                    st.metric(
                        label=f"{emoji} {disease}",
                        value=f"{result['confidence']*100:.1f}%",
                        help=f"Confidence: {result['confidence']*100:.1f}%"
                    )

            st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)

            st.subheader("Heatmap Visualizations")

            cols = st.columns(3)
            heatmap_names = ["Pneumonia", "Tuberculosis", "Fibrosis"]
            for col, name in zip(cols, heatmap_names):
                with col:
                    st.image(gradcam_maps[name], use_container_width=True)
                    st.markdown(f"<p style='text-align: center; font-weight: 600;'>{name} Heatmap</p>", unsafe_allow_html=True)
                    
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")

    else:
        st.info("ℹ️ Please upload a chest X-ray image to get started.")

    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)
    st.markdown(
        "<p style='text-align:center; font-weight: 600;'>"
        "Built with ❤️ using Streamlit & PyTorch</p>",
        unsafe_allow_html=True
    )

# === ABOUT PAGE ===
elif page == "About":
    
    st.markdown("""
    <style>
    
    .about-box {
        background: linear-gradient(to right, #0B3C51, #0B3C51);
        padding: 60px 20px;
        border-radius: 8px;
        text-align: center;
        color: white;
    }
    
    </style>

    <div class="about-box">
        <h1>About RespiraScan</h1>
        <p style="font-size: 20px;">
            AI-powered diagnostic assistant for detecting lung diseases from chest X-rays
        </p>
    </div>
""", unsafe_allow_html=True)


    st.markdown("<h2 style='text-align:center; '>🧠 Model Performance</h2>", unsafe_allow_html=True)
    perf_cols = st.columns(3)
    stats = [
        {
            "name": "Pneumonia Detection",
            "value": "99.62%",
            "emoji": "🫁",
            "samples": "3,166 balanced images",
            "color": "#1f77b4"
        },
        {
            "name": "Tuberculosis Detection",
            "value": "99.93%",
            "emoji": "🧬",
            "samples": "1,400 balanced images",
            "color": "#2ca02c"
        },
        {
            "name": "Fibrosis Detection",
            "value": "99.31%",
            "emoji": "🌫️",
            "samples": "1,454 balanced images",
            "color": "#9467bd"
        },
    ]

    for col, metric in zip(perf_cols, stats):
        with col:
            st.markdown(f"""
            <div style='
                background-color: white;
                border-radius: 12px;
                padding: 20px;
                margin: 10px 5px;
                box-shadow: 4px 4px 0px #0c4a6e;
                display: flex;
                text-align: center;
                flex-direction: column;
                justify-content: space-between;
            '>
                <div style='font-size: 40px; color: {metric['color']}; margin-bottom: 10px;'>{metric['emoji']}</div>
                <div style='font-size: 26px; font-weight: bold; color: black;'>{metric['value']}</div>
                <div style='font-size: 18px; font-weight: 600; color: #333;'>{metric['name']}</div>
                <div style='font-size: 14px; color: #666;font-weight: 600; margin-top: 8px;'>{metric['samples']}</div>
            </div>
        """, unsafe_allow_html=True)
        
    

    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)    
    st.markdown("<h2 style='text-align:center; '>Datasets Used</h2>", unsafe_allow_html=True)
    dataset_info = [
    {
        "name": "Lung Disease Dataset (4 Types)",
        "source": "Kaggle (omkarmanohardalvi)",
        "samples": "6,423 images",
        "purpose": "General disease classification"
    },
    {
        "name": "Chest X-ray Pneumonia",
        "source": "Kaggle (paultimothymooney)",
        "samples": "5,863 images",
        "purpose": "Pneumonia detection"
    },
    {
        "name": "TB Chest X-ray Dataset",
        "source": "Kaggle (tawsifurrahman)",
        "samples": "3,500 images",
        "purpose": "Tuberculosis detection"
    },
    {
        "name": "Pulmonary Fibrosis",
        "source": "Kaggle (aryashetty29)",
        "samples": "1,454 images",
        "purpose": "Fibrosis detection"
    }
    ]
    cols = st.columns(4)
    for col, data in zip(cols, dataset_info):
        with col:
            st.markdown(f"""
            <div style='
                background-color: white;
                border-radius: 12px;
                padding: 20px;
                margin: 10px 5px;
                box-shadow: 4px 4px 0px #0c4a6e;
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                color: black;
            '>
                <div style='font-weight: 700; font-size: 17px;'>{data['name']}</div>
                <div style='margin-top: 8px; color: #333; font-weight: 600; font-size: 15px;'>Source: <i>{data['source']}</i></div>
                <div style='margin-top: 4px; font-size: 15px;'>Samples: {data['samples']}</div>
                <div style='margin-top: 4px; font-size: 15px;'>Used for: {data['purpose']}</div>
            </div>
        """, unsafe_allow_html=True)


    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)    
    st.markdown("""
    <style>
.section-title {
    text-align: center;
    font-size: 2rem; /* scalable font size */
    font-weight: bold;
    margin-bottom: 1.875rem; /* 30px */
}
.subsection-title {
    font-size: 1.375rem; /* 22px */
    font-weight: bold;
    margin: 1.25rem 0 0.9375rem; /* 20px 0 15px */
    display: flex;
    align-items: center;
}
.card {
    background-color: white;
    padding: 0.9375rem 1.25rem; /* 15px 20px */
    border-radius: 12px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.1);
    display: flex;
    align-items: center;
    margin-bottom: 1.25rem; /* 20px */
    max-width: 100%;
    width: 100%;
    box-sizing: border-box;
}
.card-icon {
    font-size: 1.75rem; /* 28px */
    margin-right: 0.9375rem; /* 15px */
    flex-shrink: 0;
}
.card-content {
    display: flex;
    flex-direction: column;
    flex: 1; /* take full remaining space */
}
.card-title {
    font-size: 0.875rem; /* 14px */
    margin: 0;
}
.card-value {
    font-size: 1.125rem; /* 18px */
    font-weight: bold;
    color: #111827;
    margin-top: 0.125rem; /* 2px */
}
ul {
    padding-left: 1.25rem; /* 20px */
    font-size: 1rem; /* 16px */
    margin-top: 0;
}

/* Make the columns responsive */
[data-testid="stHorizontalBlock"] {
    display: flex !important;
    flex-wrap: wrap;  /* allow wrapping */
    gap: 1.25rem; /* space between columns */
}

/* Override the default fixed width of columns */
[data-testid="stVerticalBlock"] {
    flex: 1 1 300px; /* grow, shrink, basis 300px minimum */
    max-width: 100%;
    box-sizing: border-box;
}

/* Cards inside columns will take full width */
[data-testid="stVerticalBlock"] .card {
    width: 100%;
}

/* Responsive text sizing on small screens */
@media (max-width: 600px) {
    .section-title {
        font-size: 1.5rem;
        margin-bottom: 1rem;
    }
    .subsection-title {
        font-size: 1.125rem;
        margin: 1rem 0 0.75rem;
    }
    .card-icon {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    .card-title {
        font-size: 0.8rem;
    }
    .card-value {
        font-size: 1rem;
    }
    ul {
        font-size: 0.9rem;
    }
}
</style>
""", unsafe_allow_html=True)



    # Main Title
    st.markdown("<div class='section-title'>Model Architecture</div>", unsafe_allow_html=True)

    # Two Columns
    col1, col2 = st.columns(2)

    # Left: Technical Specs Cards
    with col1:
        st.markdown("<div class='subsection-title'>🧪 Technical Specifications</div>", unsafe_allow_html=True)
        specs = [
        {"icon": "💻", "title": "Base Architecture", "value": "DenseNet121"},
        {"icon": "🧪", "title": "Framework", "value": "PyTorch"},
        {"icon": "🖼️", "title": "Input Size", "value": "224×224px"},
        {"icon": "⏱️", "title": "Training Time", "value": "30 epochs"},
        ]
        for i in range(0, len(specs), 2):
            cols = st.columns(2)
            for j in range(2):
                if i + j < len(specs):
                    spec = specs[i + j]
                    with cols[j]:
                        st.markdown(
                            f"""
                            <div class="card">
                                <div class="card-icon">{spec['icon']}</div>
                                <div class="card-content">
                                    <div class="card-title">{spec['title']}</div>
                                    <div class="card-value">{spec['value']}</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True
                        )

        
    # Right: Preprocessing Steps
    with col2:
        st.markdown("<div class='subsection-title'>⚖️ Preprocessing Steps</div>", unsafe_allow_html=True)
        st.markdown("""
        <ul>
            <li>Image resizing to 224×224</li>
            <li>Normalization (μ=[0.485, 0.456, 0.406], σ=[0.229, 0.224, 0.225])</li>
            <li>Data augmentation (rotation, flipping, color jitter)</li>
            <li>Class balancing (undersampling majority classes)</li>
            <li>Train/Val split (80/20)</li>
        </ul>
        """, unsafe_allow_html=True)

    
    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)    # CSS Styling

    st.markdown("""
<style>
.section-title {
    text-align: center;
    font-size: 32px;
    font-weight: bold;
    margin-bottom: 30px;
}
.card-row {
    display: flex;
    justify-content: center;
    gap: 30px;
    flex-wrap: wrap;
    margin-bottom: 40px;
}
.card {
    background-color: white;
    padding: 20px 25px;
    border-radius: 15px;
    box-shadow: 4px 4px 0px #0c4a6e;
    width: 100%;
    max-width: 270px;
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
    transition: transform 0.2s ease-in-out;
}
.card:hover {
    transform: translateY(-5px);
}
.card-icon {
    font-size: 36px;
    margin-bottom: 12px;
}
.card-title {
    font-weight: bold;
    font-size: 18px;
    color: #111827;
    margin-bottom: 6px;
}
.card-desc {
    font-size: 15px;
    color: #4b5563;
    font-weight: 600;
}

/* Make cards stack nicely on smaller screens */
@media (max-width: 768px) {
    .card {
        max-width: 90%;
    }
}
</style>
""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Clinical Impact</div>", unsafe_allow_html=True)

    st.markdown("""
<div class="card-row">
    <div class="card">
        <div class="card-icon">🏥</div>
        <div class="card-title">Early Detection</div>
        <div class="card-desc">Identifies diseases at stages when treatment is most effective</div>
    </div>
    <div class="card">
        <div class="card-icon">🛡️</div>
        <div class="card-title">Reduced Workload</div>
        <div class="card-desc">Helps radiologists prioritize urgent cases</div>
    </div>
    <div class="card">
        <div class="card-icon">📱</div>
        <div class="card-title">Accessibility</div>
        <div class="card-desc">Potential for deployment in resource-limited settings</div>
    </div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)


# === Future Directions ===
    st.markdown("<div class='section-title'>Future Directions</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card-row">
        <div class="card">
            <div class="card-desc">Expand to additional pulmonary conditions</div>
        </div>
        <div class="card">
            <div class="card-desc">Incorporate 3D CT scan analysis</div>
        </div>
        <div class="card">
            <div class="card-desc">Develop mobile application for field use</div>
        </div>
        <div class="card">
            <div class="card-desc">Obtain regulatory approvals</div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Footer (optional)
    st.markdown("<hr class='custom-hr'>", unsafe_allow_html=True)      
    st.markdown(
    "<p style='text-align:center;'>© 2025 RespiraScan — Developed with ❤️ using Streamlit, PyTorch, and Grad-CAM</p>",
    unsafe_allow_html=True
)

# End section
    st.markdown("</div>", unsafe_allow_html=True)
