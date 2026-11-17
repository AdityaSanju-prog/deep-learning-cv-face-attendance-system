"""
DeepFaceAttend: Deep Learning Face Scan Attendance System
=========================================================
Flagship Streamlit Web Application
Seamlessly integrates:
- Unit I: Image Foundations & Preprocessing
- Unit II: CNN Architectures & Training
- Unit III: Transfer Learning (MobileNetV2) & Metric Face Embeddings
- Unit IV: Single-Stage SSD Face Detection with NMS
- Unit V: Segmentation & Vision Transformers
- Unit VI: Grad-CAM Explainability & Passive Anti-Spoofing Liveness Detection
"""

import os
import io
import time
from datetime import date, datetime
import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="DeepFaceAttend | DL Face Biometrics System",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling for modern academic UI
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .status-badge-present {
        background-color: #DCFCE7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
    .status-badge-spoof {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 4px 10px;
        border-radius: 9999px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_attendance_pipeline():
    """Caches the heavy deep learning models and database pipeline."""
    from core.attendance_engine import AttendancePipeline
    pipeline = AttendancePipeline()
    return pipeline


# Initialize Pipeline
with st.spinner("Initializing Deep Neural Networks (MobileNetV2, SSD Face Detector, Grad-CAM)..."):
    pipeline = load_attendance_pipeline()

# =====================================================================
# Sidebar Navigation & Diagnostics
# =====================================================================
st.sidebar.image("https://img.icons8.com/color/96/facial-recognition.png", width=80)
st.sidebar.title("DeepFaceAttend")
st.sidebar.caption("Computer Vision & Deep Learning Attendance System")

menu = st.sidebar.radio(
    "Navigation Menu",
    [
        "📸 Live Attendance Scanner",
        "👤 Student Registration",
        "📊 Attendance Logs & Analytics",
        "🔬 Model Explainability (Grad-CAM)",
        "🧠 Syllabus & Architecture Hub"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Pipeline Configuration")
sim_threshold = st.sidebar.slider(
    "Match Confidence Threshold",
    min_value=0.40,
    max_value=0.90,
    value=0.60,
    step=0.02,
    help="Minimum cosine similarity required between query face and enrolled template."
)

st.sidebar.markdown("---")
gallery_count = len(pipeline.gallery)
today_stats = pipeline.db.get_attendance_stats()
st.sidebar.markdown(f"**Enrolled Students:** `{gallery_count}`")
st.sidebar.markdown(f"**Marked Today:** `{today_stats['present_count']} / {gallery_count}`")
st.sidebar.markdown(f"**Attendance Rate:** `{today_stats['attendance_rate']}%`")

# Helper to convert PIL / Uploaded to OpenCV BGR
def pil_to_cv2(pil_img: Image.Image) -> np.ndarray:
    rgb = np.array(pil_img.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


# =====================================================================
# Page 1: Live Attendance Scanner
# =====================================================================
if menu == "📸 Live Attendance Scanner":
    st.markdown('<div class="main-title">📸 Live Face Scan Attendance Scanner</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Real-time face detection (Unit IV), liveness check (Unit VI), and deep metric recognition (Unit III)</div>', unsafe_allow_html=True)

    tab_cam, tab_upload = st.tabs(["📷 Webcam Scan", "📁 Upload Image / Classroom Photo"])

    # --- TAB A: Webcam Scan ---
    with tab_cam:
        st.info("💡 Position your face in front of the camera and click **Take Photo** to verify identity and record attendance.")
        cam_photo = st.camera_input("Attendance Scanner Camera")

        if cam_photo is not None:
            image_pil = Image.open(cam_photo)
            frame_bgr = pil_to_cv2(image_pil)

            with st.spinner("Processing facial biometrics & checking liveness..."):
                annotated_bgr, events = pipeline.process_frame(
                    frame_bgr, mark_attendance=True, threshold=sim_threshold
                )

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            col1, col2 = st.columns([3, 2])

            with col1:
                st.image(annotated_rgb, caption="Annotated Live Scanner Feed", use_container_width=True)

            with col2:
                st.subheader("Verification Results")
                if not events:
                    st.warning("No face detected. Please ensure proper ambient lighting and face the camera directly.")
                else:
                    for ev in events:
                        status = ev["status"]
                        if status == "Present":
                            st.success(f"✅ **{ev['name']}** ({ev['roll_number']})")
                            st.caption(f"Dept: {ev['department']} | Match: {ev['confidence']*100:.1f}% | Liveness: {ev['liveness_score']}")
                            st.toast(f"🎉 Attendance marked for {ev['name']}!", icon="✅")
                        elif status == "Already Marked":
                            st.info(f"ℹ️ **{ev['name']}** ({ev['roll_number']})")
                            st.caption(f"Status: Already checked-in today | Match: {ev['confidence']*100:.1f}%")
                        elif status == "Spoof":
                            st.error(f"🚨 **SPOOF ALERT**: Potential Presentation Attack Detected!")
                            st.caption(f"Liveness Confidence: {ev['liveness_score']} (Failed threshold)")
                        else:
                            st.warning(f"⚠️ **Unknown Individual**")
                            st.caption(f"Best Match Score: {ev['confidence']*100:.1f}% (Below threshold {sim_threshold*100:.0f}%)")

    # --- TAB B: Upload Photo ---
    with tab_upload:
        st.info("Upload single or group classroom photos to test multi-face detection, liveness, and batch attendance marking.")
        uploaded_file = st.file_uploader("Upload Image (JPG, PNG)", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image_pil = Image.open(uploaded_file)
            frame_bgr = pil_to_cv2(image_pil)

            with st.spinner("Running Deep Face Detection & Verification..."):
                annotated_bgr, events = pipeline.process_frame(
                    frame_bgr, mark_attendance=True, threshold=sim_threshold
                )

            annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
            col1, col2 = st.columns([3, 2])

            with col1:
                st.image(annotated_rgb, caption="Scanned Image Results", use_container_width=True)

            with col2:
                st.subheader(f"Faces Detected: {len(events)}")
                if events:
                    df_events = pd.DataFrame([
                        {
                            "Name": e["name"],
                            "Roll No": e["roll_number"],
                            "Department": e["department"],
                            "Match Score": f"{e['confidence']*100:.1f}%",
                            "Liveness": e["liveness_score"],
                            "Status": e["status"]
                        }
                        for e in events
                    ])
                    st.dataframe(df_events, use_container_width=True)


# =====================================================================
# Page 2: Student Registration & Enrollment
# =====================================================================
elif menu == "👤 Student Registration":
    st.markdown('<div class="main-title">👤 Student Biometric Enrollment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Register students, extract 512-D deep face embeddings (Unit III), and persist to SQLite database</div>', unsafe_allow_html=True)

    col_form, col_cam = st.columns([1, 1])

    with col_form:
        st.subheader("Student Details")
        reg_roll = st.text_input("Roll / Registration Number*", placeholder="e.g. 21CS042").strip().upper()
        reg_name = st.text_input("Full Name*", placeholder="e.g. Aditya Sharma").strip()
        reg_dept = st.selectbox("Department*", [
            "Computer Science & Engineering",
            "Artificial Intelligence & Data Science",
            "Information Technology",
            "Electronics & Communication",
            "Mechanical Engineering",
            "Electrical Engineering"
        ])

        source_type = st.radio("Face Capture Source", ["Camera Snapshot", "Upload Passport Photo"], horizontal=True)

    with col_cam:
        st.subheader("Face Capture")
        reg_bgr = None

        if source_type == "Camera Snapshot":
            reg_cam = st.camera_input("Capture Student Face Photo")
            if reg_cam is not None:
                reg_bgr = pil_to_cv2(Image.open(reg_cam))
        else:
            reg_upload = st.file_uploader("Upload Clear Face Photo", type=["jpg", "jpeg", "png"], key="reg_upload")
            if reg_upload is not None:
                reg_bgr = pil_to_cv2(Image.open(reg_upload))

    st.markdown("---")
    if st.button("🚀 Enroll Student Face", type="primary", use_container_width=True):
        if not reg_roll or not reg_name:
            st.error("Please provide both Roll Number and Full Name.")
        elif reg_bgr is None:
            st.error("Please capture or upload a clear photo of the student's face.")
        else:
            with st.spinner("Extracting 512-D MobileNetV2 embedding and updating gallery..."):
                ok, msg, face_crop = pipeline.register_new_student(
                    reg_bgr, roll_number=reg_roll, name=reg_name, department=reg_dept
                )

            if ok:
                st.success(f"🎉 {msg}")
                if face_crop is not None:
                    col1, col2 = st.columns([1, 3])
                    with col1:
                        st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), caption="Normalized Face Crop (112x112)", width=150)
                    with col2:
                        st.markdown(f"""
                        **Enrolled Profile:**
                        - **Name:** {reg_name}
                        - **Roll Number:** {reg_roll}
                        - **Department:** {reg_dept}
                        - **Feature Representation:** 512-D L2-Normalized Unit Vector
                        """)
            else:
                st.error(f"Enrollment Error: {msg}")

    # Enrolled Student Directory
    st.markdown("---")
    st.subheader(f"📋 Enrolled Students Directory ({len(pipeline.gallery)})")
    students = pipeline.db.get_all_students()
    if students:
        df_students = pd.DataFrame(students)[["roll_number", "name", "department", "created_at"]]
        df_students.columns = ["Roll Number", "Full Name", "Department", "Enrolled On"]
        st.dataframe(df_students, use_container_width=True)
    else:
        st.info("No students enrolled yet. Use the form above to add the first student!")


# =====================================================================
# Page 3: Attendance Logs & Analytics
# =====================================================================
elif menu == "📊 Attendance Logs & Analytics":
    st.markdown('<div class="main-title">📊 Attendance Logs & Department Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Review daily check-in records, attendance statistics, and export official CSV reports</div>', unsafe_allow_html=True)

    col_date, col_export = st.columns([2, 1])
    with col_date:
        selected_date = st.date_input("Select Date for Report", value=date.today())
        selected_date_str = selected_date.isoformat()

    stats = pipeline.db.get_attendance_stats(selected_date_str)

    # Top KPI Metrics Cards
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Enrolled", stats["total_students"])
    m2.metric("Present Today", stats["present_count"], delta=f"{stats['attendance_rate']}%")
    m3.metric("Absent Today", stats["absent_count"])
    m4.metric("Attendance Rate", f"{stats['attendance_rate']}%")

    st.markdown("---")
    records = pipeline.db.get_attendance_by_date(selected_date_str)

    if records:
        df_records = pd.DataFrame(records)
        df_display = df_records[["roll_number", "name", "department", "timestamp", "confidence", "liveness_score", "status"]]
        df_display.columns = ["Roll No", "Full Name", "Department", "Check-In Time", "Match Score", "Liveness Score", "Status"]

        st.subheader(f"Attendance Records for {selected_date_str}")
        st.dataframe(df_display, use_container_width=True)

        # Department Breakdown Chart
        st.subheader("Department Attendance Distribution")
        dept_counts = df_display["Department"].value_counts().reset_index()
        dept_counts.columns = ["Department", "Present Count"]
        st.bar_chart(dept_counts.set_index("Department"), color="#2563EB")

        # CSV Export
        csv_buffer = io.StringIO()
        df_display.to_csv(csv_buffer, index=False)
        st.download_button(
            label="📥 Download Daily Report (CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"attendance_report_{selected_date_str}.csv",
            mime="text/csv"
        )
    else:
        st.warning(f"No attendance marked yet for {selected_date_str}.")


# =====================================================================
# Page 4: Model Explainability (Grad-CAM)
# =====================================================================
elif menu == "🔬 Model Explainability (Grad-CAM)":
    st.markdown('<div class="main-title">🔬 Model Explainability & Grad-CAM Visualizer</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Unit VI: Gradient-weighted Class Activation Mapping (Grad-CAM) for Deep CNN feature interpretability</div>', unsafe_allow_html=True)

    st.markdown(r"""
    > **Mathematical Principle (Unit VI)**:
    > Grad-CAM computes gradients of the target identity score with respect to the feature activation maps of the final convolutional layer:
    > $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
    > The resulting saliency heatmap highlights which specific facial structures (eyes, bridge of nose, jawline) the deep neural network focused on to verify the student's identity.
    """)

    opt = st.radio("Choose Input Mode", ["Select Enrolled Student", "Upload Custom Face Image"], horizontal=True)
    target_face_bgr = None

    if opt == "Select Enrolled Student":
        students = pipeline.db.get_all_students()
        if not students:
            st.info("No enrolled students available. Please enroll a student first.")
        else:
            selected_stud = st.selectbox("Select Student", students, format_func=lambda s: f"{s['name']} ({s['roll_number']})")
            if selected_stud and selected_stud.get("photo_path") and os.path.exists(selected_stud["photo_path"]):
                target_face_bgr = cv2.imread(selected_stud["photo_path"])
    else:
        up = st.file_uploader("Upload Face Image for Grad-CAM Analysis", type=["jpg", "jpeg", "png"])
        if up is not None:
            target_face_bgr = pil_to_cv2(Image.open(up))

    if target_face_bgr is not None:
        with st.spinner("Computing backward gradients and synthesizing Grad-CAM heatmap..."):
            face_resized, heatmap, overlay = pipeline.explain_face(target_face_bgr)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB), caption="1. Input Face Crop (224x224)", use_container_width=True)
        with c2:
            st.image(heatmap, caption="2. Saliency Activation Heatmap (Grad-CAM)", use_container_width=True, clamp=True)
        with c3:
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="3. Blended Explainability Overlay", use_container_width=True)

        st.success("✅ **Interpretation**: Warm red/yellow zones represent salient discriminative facial landmarks utilized by the network.")


# =====================================================================
# Page 5: Syllabus & Architecture Hub
# =====================================================================
elif menu == "🧠 Syllabus & Architecture Hub":
    st.markdown('<div class="main-title">🧠 Deep Learning in Computer Vision: Curriculum Hub</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Exhaustive academic syllabus mapping, mathematical formulas, and Viva/Exam review for Units I to VI</div>', unsafe_allow_html=True)

    with st.expander("📘 UNIT I: Foundations of Computer Vision and Deep Learning", expanded=True):
        st.markdown(r"""
        ### Key Concepts & Project Alignment
        - **Role of CV in AI**: Enables machine perception from raw photon matrices to semantic biometric identities.
        - **Traditional vs Deep Learning**:
          - *Traditional*: Handcrafted features (Haar cascades, HOG descriptors) + SVM. Brittle to illumination and occlusions.
          - *Deep Learning*: End-to-end hierarchical representation learning where convolutional kernels discover optimal filters automatically.
        - **Image Fundamentals**:
          - Pixels, channels (RGB vs BGR vs Grayscale vs HSV).
          - Resizing with aspect-ratio preservation (Letterbox padding) to prevent geometric distortion.
          - Normalization: ImageNet standardization $Z = \frac{X - \mu}{\sigma}$ with $\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$.
        - **Neural Network Basics**:
          - Perceptron learning rule: $w \leftarrow w + \eta (y - \hat{y}) x$.
          - Multi-Layer Perceptrons (MLP), activations (ReLU, LeakyReLU, Sigmoid, Softmax), Cross-Entropy Loss, and Adam/SGD optimizers.
        - **Project File**: `modules/unit1_foundations.py`
        """)

    with st.expander("📙 UNIT II: Convolutional Neural Networks and Training Techniques", expanded=False):
        st.markdown(r"""
        ### Key Concepts & Project Alignment
        - **Convolution Operation**:
          - Output spatial dimension formula:
            $$O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$
          - Receptive Field calculation: $RF_{l} = RF_{l-1} + (K_l - 1) \times \text{stride}_{cumulative}$.
        - **Modern CNN Architectures Evolution**:
          - **LeNet-5 (1998)**: Conv-Pool-FC baseline for handwritten digits.
          - **AlexNet (2012)**: ReLU, Dropout, multi-GPU training.
          - **VGG (2014)**: Factorized $3 \times 3$ conv stacks replacing large $7 \times 7$ kernels.
          - **Inception (2014)**: Multi-scale parallel kernels ($1\times 1, 3\times 3, 5\times 5$) and bottleneck reduction.
          - **ResNet (2015)**: Identity residual skip connections $H(x) = F(x) + x$ solving vanishing gradients in deep networks.
        - **Regularization & Training Dynamics**:
          - Batch Normalization ($z = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \gamma + \beta$) and Dropout ($p=0.4$).
          - Data Augmentation (Random flips, rotation $\pm 10^\circ$, color jitter, Gaussian blur).
          - Weight Initialization: Kaiming (He) normal for ReLU and Xavier (Glorot) uniform for linear heads.
          - Cosine Annealing learning rate scheduling and Early Stopping.
        - **Project File**: `modules/unit2_cnn_training.py`
        """)

    with st.expander("📗 UNIT III: Transfer Learning and Fine-Tuning for Vision Tasks", expanded=False):
        st.markdown(r"""
        ### Key Concepts & Project Alignment
        - **Pre-trained Backbone**: MobileNetV2 with depthwise separable convolutions and inverted residual linear bottlenecks (3.5M parameters).
        - **Transfer Learning Strategies**:
          1. *Feature Extraction*: Freezing all backbone layers (`requires_grad = False`), training only the projection head.
          2. *Fine-Tuning*: Unfreezing the top residual blocks with discriminative layer-wise learning rates (e.g. $10^{-5}$ for backbone, $10^{-3}$ for metric head).
        - **Deep Metric Learning**:
          - Projects face crops to a 512-dimensional $L_2$-normalized unit hypersphere ($\|\mathbf{u}\|_2 = 1.0$).
          - **Cosine Similarity**: $\cos(\theta) = \mathbf{u} \cdot \mathbf{v}$.
          - Threshold-based biometric verification ($\ge 0.60$ for positive identification).
        - **Project File**: `modules/unit3_transfer_learning.py`
        """)

    with st.expander("📕 UNIT IV: Object Detection and Localization", expanded=False):
        st.markdown(r"""
        ### Key Concepts & Project Alignment
        - **Detection Principles**: Bounding box coordinate systems $(x_{min}, y_{min}, x_{max}, y_{max})$ vs $(x, y, w, h)$.
        - **Intersection over Union (IoU)**:
          $$\text{IoU} = \frac{\text{Area}(B_1 \cap B_2)}{\text{Area}(B_1 \cup B_2)}$$
        - **Non-Maximum Suppression (NMS)**:
          - Greedy algorithm sorting proposals by confidence score, iteratively discarding candidates with $\text{IoU} > \tau_{NMS}$.
        - **Detection Pipelines**:
          - *Two-Stage*: R-CNN, Fast R-CNN, Faster R-CNN (Region Proposal Network).
          - *Single-Stage*: SSD (Single-Shot MultiBox Detector), YOLO variants, RetinaNet with Focal Loss.
        - **Production Detector**: OpenCV DNN SSD with ResNet-10 backbone running at $300 \times 300$ resolution with sub-15ms CPU latency.
        - **Project File**: `modules/unit4_object_detection.py`
        """)

    with st.expander("📓 UNIT V: Image Segmentation and Advanced Vision Architectures", expanded=False):
        st.markdown(r"""
        ### Key Concepts & Project Alignment
        - **Segmentation Paradigms**:
          - *Semantic Segmentation*: Pixel-level class labeling without individual object separation.
          - *Instance Segmentation*: Mask R-CNN predicting both bounding box, class, and binary pixel mask.
        - **UNet Architecture**: Contracting encoder path + expanding decoder path with lateral skip connections preserving spatial details.
        - **Vision Transformers (ViT)**:
          - Eliminates convolutions in favor of Self-Attention across $16 \times 16$ image patches.
          - Scaled Dot-Product Attention:
            $$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right) V$$
          - Multi-Head Self-Attention (MHSA) + learnable `[CLS]` token and 1D positional encodings.
          - Swin Transformer: Hierarchical shifted-window self-attention.
        - **Project File**: `modules/unit5_segmentation_transformers.py`
        """)

    with st.expander("📔 UNIT VI: Generative Vision Models and Interpretability", expanded=False):
        st.markdown(r"""
        ### Key Concepts & Project Alignment
        - **Generative Deep Learning**:
          - *VAEs*: Latent probabilistic manifold with the Reparameterization Trick ($z = \mu + \sigma \odot \epsilon$) maximizing ELBO.
          - *GANs*: Minimax game between Generator $G(z)$ and Discriminator $D(x)$; DCGAN, CycleGAN (unpaired translation), StyleGAN (AdaIN style modulation).
        - **Vision-Language Alignment**: CLIP dual-encoder contrastive learning (InfoNCE).
        - **Model Interpretability (Grad-CAM)**:
          - Global average pooling of gradients from the target identity with respect to the last convolutional feature maps:
            $$\alpha_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{ij}^k}, \quad L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_k \alpha_k^c A^k \right)$$
        - **Biometric Security (Anti-Spoofing)**:
          - Passive liveness detection using Laplacian spatial variance and 2D Fast Fourier Transform (FFT) high-frequency power distribution to defeat paper photo and screen replay attacks.
        - **Project File**: `modules/unit6_explainability_generative.py`
        """)

st.sidebar.markdown("---")
st.sidebar.caption("DeepFaceAttend v1.0 | Academic AI Project")
