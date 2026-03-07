"""
🌳 Tree Classification App
Multi-stage hierarchical AI pipeline for tree species and growth stage detection.
Uses 4 trained models + Grad-CAM heatmaps.
"""

import streamlit as st
from PIL import Image
import torch
import numpy as np
from utils.model_loader import load_all_models
from utils.predictor import predict_tree, predict_species, predict_stage
from utils.gradcam import generate_gradcam

# ──────────────────────────────────────────────
# PAGE CONFIGURATION
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="🌳 Tree Classifier",
    page_icon="🌳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# CUSTOM CSS STYLING
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        color: #2d6a4f;
        text-align: center;
        padding: 1rem 0;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #52796f;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background: #f0f7f4;
        border-left: 5px solid #2d6a4f;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 1rem 1.5rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    .stage-badge {
        display: inline-block;
        background: #2d6a4f;
        color: white;
        padding: 0.3rem 1rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown('<div class="main-title">🌳 Tree Species & Growth Stage Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a tree photo to identify its species and growth stage using AI</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────
# SIDEBAR — INFO PANEL
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("📖 How It Works")
    st.markdown("""
    This app uses **4 AI models** in sequence:

    1. 🔍 **Tree Detection** — Is it a tree?
    2. 🌿 **Species Detection** — Mango or White Gum?
    3. 📏 **Stage Classification** — What growth stage?

    Each step also shows a **Grad-CAM heatmap** — a visualization of which part of the image the AI focused on.
    """)

    st.divider()
    st.header("⚠️ System Limitations")
    st.markdown("""
    This system is currently trained only on:
    - 🥭 **Mango Trees**
    - 🌿 **White Gum / Eucalyptus Trees**

    Upload images of these species for accurate results.
    """)

    st.divider()
    st.header("📊 Growth Stages")
    st.markdown("""
    - 🌱 **Seedling** — Very young, just sprouted
    - 🌿 **Sapling** — Young, growing
    - 🌳 **Mature** — Fully grown
    - 🍂 **Overmature** — Past peak, aging
    """)

# ──────────────────────────────────────────────
# MODEL LOADING (cached so it only runs once)
# ──────────────────────────────────────────────
@st.cache_resource
def get_models():
    """Load all 4 models once and cache them in memory."""
    with st.spinner("Loading AI models... (this only happens once)"):
        models = load_all_models()
    return models

try:
    models = get_models()
    st.sidebar.success("✅ All models loaded successfully")
except Exception as e:
    st.error(f"❌ Failed to load models: {e}")
    st.info("Make sure your model files are in the `models/` folder.")
    st.stop()

# ──────────────────────────────────────────────
# IMAGE UPLOAD
# ──────────────────────────────────────────────
st.divider()
uploaded_file = st.file_uploader(
    "📤 Upload a tree image (JPG, PNG, JPEG)",
    type=["jpg", "jpeg", "png"],
    help="For best results, use a clear photo of a single tree."
)

if uploaded_file is None:
    # Show a friendly placeholder when no image is uploaded
    st.info("👆 Upload an image above to start the classification pipeline.")
    st.markdown("""
    ### 🧭 What happens after you upload:
    | Step | What the AI does |
    |------|-----------------|
    | Step 1 | Checks if the image contains a tree |
    | Step 2 | Identifies if it's a Mango or White Gum tree |
    | Step 3 | Predicts the tree's growth stage |
    | All steps | Shows a Grad-CAM heatmap highlighting AI focus areas |
    """)
    st.stop()

# ──────────────────────────────────────────────
# IMAGE DISPLAY
# ──────────────────────────────────────────────
image = Image.open(uploaded_file).convert("RGB")

col_img, col_info = st.columns([1, 1])
with col_img:
    st.subheader("📷 Uploaded Image")
    st.image(image, caption="Your uploaded image", use_container_width=True)
with col_info:
    st.subheader("🔬 Analysis Pipeline")
    st.markdown("""
    The image will be processed through the following stages:

    ```
    Image
      │
      ▼
    [1] Tree vs Non-Tree
      │ (if tree detected)
      ▼
    [2] Species: Mango or White Gum?
      │
      ▼
    [3] Growth Stage Classification
    ```
    Each step outputs a **confidence score** and a **Grad-CAM heatmap**.
    """)

# ──────────────────────────────────────────────
# PIPELINE EXECUTION
# ──────────────────────────────────────────────
st.divider()
st.header("🤖 AI Classification Results")

with st.spinner("🔍 Running analysis..."):

    # ── PHASE 1: Tree vs Non-Tree ──────────────────
    st.subheader("Phase 1 — Tree Detection")

    try:
        tree_label, tree_conf, tree_tensor = predict_tree(image, models["tree_vs_nontree"])
        tree_heatmap = generate_gradcam(image, models["tree_vs_nontree"], tree_tensor)
    except Exception as e:
        st.error(f"Error in tree detection: {e}")
        st.stop()

    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Original Image", use_container_width=True)
    with col2:
        st.image(tree_heatmap, caption="Grad-CAM: Where the model looked", use_container_width=True)

    if tree_label == "Non-Tree":
        st.markdown(f"""
        <div class="result-box">
        ❌ <b>Result:</b> Non-Tree detected ({tree_conf:.1%} confidence)<br>
        The pipeline has stopped. Please upload an image containing a tree.
        </div>
        """, unsafe_allow_html=True)
        st.stop()
    else:
        st.markdown(f"""
        <div class="result-box">
        ✅ <b>Result:</b> Tree detected ({tree_conf:.1%} confidence) — Proceeding to species detection.
        </div>
        """, unsafe_allow_html=True)

    # ── PHASE 2: Species Detection ──────────────────
    st.divider()
    st.subheader("Phase 2 — Species Detection")

    st.markdown("""
    <div class="warning-box">
    ⚠️ <b>Note:</b> This system is trained only on <b>Mango</b> and <b>White Gum (Eucalyptus)</b> trees.
    Please upload images of these species for accurate results.
    </div>
    """, unsafe_allow_html=True)

    try:
        species_label, species_conf, species_tensor = predict_species(image, models["species"])
        species_heatmap = generate_gradcam(image, models["species"], species_tensor)
    except Exception as e:
        st.error(f"Error in species detection: {e}")
        st.stop()

    col3, col4 = st.columns(2)
    with col3:
        st.image(image, caption="Original Image", use_container_width=True)
    with col4:
        st.image(species_heatmap, caption="Grad-CAM: Species focus area", use_container_width=True)

    species_icon = "🥭" if species_label == "Mango" else "🌿"
    st.markdown(f"""
    <div class="result-box">
    {species_icon} <b>Detected Species:</b> {species_label} ({species_conf:.1%} confidence)
    </div>
    """, unsafe_allow_html=True)

    # ── PHASE 3: Growth Stage Classification ────────
    st.divider()
    st.subheader("Phase 3 — Growth Stage Classification")

    # Route to the correct stage model based on species
    if species_label == "Mango":
        stage_model = models["mango_stage"]
        model_name = "Mango Stage Model"
    else:
        stage_model = models["gum_stage"]
        model_name = "White Gum Stage Model"

    st.info(f"🔀 Routing to: **{model_name}**")

    try:
        stage_label, stage_conf, stage_tensor = predict_stage(image, stage_model, species_label)
        stage_heatmap = generate_gradcam(image, stage_model, stage_tensor)
    except Exception as e:
        st.error(f"Error in stage classification: {e}")
        st.stop()

    col5, col6 = st.columns(2)
    with col5:
        st.image(image, caption="Original Image", use_container_width=True)
    with col6:
        st.image(stage_heatmap, caption="Grad-CAM: Stage focus area", use_container_width=True)

    stage_icons = {"Seedling": "🌱", "Sapling": "🌿", "Mature": "🌳", "Overmature": "🍂"}
    stage_icon = stage_icons.get(stage_label, "🌲")

    st.markdown(f"""
    <div class="result-box">
    {stage_icon} <b>Growth Stage:</b> <span class="stage-badge">{stage_label}</span>
    &nbsp; ({stage_conf:.1%} confidence)
    </div>
    """, unsafe_allow_html=True)

    # ── FINAL SUMMARY ────────────────────────────────
    st.divider()
    st.subheader("📋 Summary")
    summary_col1, summary_col2, summary_col3 = st.columns(3)
    with summary_col1:
        st.metric("🔍 Detection", "Tree", f"{tree_conf:.1%}")
    with summary_col2:
        st.metric(f"{species_icon} Species", species_label, f"{species_conf:.1%}")
    with summary_col3:
        st.metric(f"{stage_icon} Growth Stage", stage_label, f"{stage_conf:.1%}")
