import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

st.set_page_config(
    page_title="Crop Disease Detector",
    page_icon="🌿",
    layout="wide"
)

st.markdown("""
<style>
    /* Import a cute, rounded Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;800&display=swap');

    /* Apply the font to the whole app */
    html, body, [class*="css"] {
        font-family: 'Nunito', sans-serif !important;
    }

    /* Hide the Streamlit top menu and footer to make it look like a standalone app */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    section.main > div {
    padding-top: 0rem;
    }

    [data-testid="stAppViewContainer"] > .main {
        padding-top: 0rem;
    }

    [data-testid="stAppViewContainer"] > .main > div {
        padding-top: 0rem;
    }

    /* Bubbly Header */
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        color: #4ade80; /* Brighter, friendly green */
        margin-bottom: 0rem;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 600;
        color: #a7f3d0; /* Soft pastel green */
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 800;
        margin-bottom: 1rem;
        color: #f3f4f6;
    }

    /* Soft, pill-shaped Prediction Card */
    .prediction-card {
        padding: 2rem;
        border-radius: 30px; /* Maximum rounding for cuteness */
        background-color: rgba(74, 222, 128, 0.15);
        border: 2px solid rgba(74, 222, 128, 0.3);
        text-align: center;
        margin-top: 0.5rem;
    }

    .prediction-label {
        color: #4ade80;
        font-size: 0.9rem;
        font-weight: 800;
        letter-spacing: 1px;
        margin-bottom: 0.5rem;
    }

    .prediction-name {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
        margin-bottom: 1rem;
    }

    .confidence-label {
        font-size: 0.9rem;
        font-weight: 600;
        color: #9ca3af;
    }

    .confidence-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: #4ade80;
    }

    /* Friendly "Waiting" State Card */
    .ready-card {
        padding: 3rem 2rem;
        border-radius: 30px;
        background-color: rgba(255, 255, 255, 0.03);
        border: 3px dashed rgba(255, 255, 255, 0.1);
        text-align: center;
        margin-top: 0.5rem;
    }

    .ready-title {
        font-size: 1.5rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
    }

    .ready-text {
        color: #9ca3af;
        font-weight: 600;
        line-height: 1.5;
    }

    .model-info {
        padding: 1rem;
        border-radius: 20px;
        background: rgba(255,255,255,0.03);
        text-align: center;
        margin-top: 1rem;
        color: #9ca3af;
        font-size: 0.85rem;
    }

    div.stButton > button {
        background-color: #22c55e !important;
        color: #ffffff !important;
        border-radius: 25px !important;
        border: none !important;
        font-weight: 700 !important;
        font-size: 1.05rem !important;
        padding: 0.6rem 1.5rem !important;
        transition: transform 0.15s ease, background-color 0.15s ease !important;
    }

    div.stButton > button:hover {
        background-color: #16a34a !important;
        transform: scale(1.02);
    }

    /* 2. Constrain Leaf Preview Height so it doesn't create huge blank space */
    div[data-testid="stImage"] img {
        max-height: 260px !important;
        object-fit: cover !important;
        border-radius: 18px !important;
    }

    /* 3. Make the Progress Bar Green */
    div[data-testid="stProgress"] > div > div > div > div {
        background-color: #22c55e !important;
        border-radius: 10px !important;
    }
    
    div[data-testid="stProgress"] > div > div {
        background-color: rgba(255, 255, 255, 0.08) !important;
        border-radius: 10px !important;
    }

    /* 4. Soften the Tip Alert Box */
    div[data-testid="stAlert"] {
        background-color: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 16px !important;
        color: #9ca3af !important;
    }
</style>
""", unsafe_allow_html=True)

#Class names
CLASS_NAMES = [
    "Maize_Northern_Leaf_Blight",
    "Tomato_Late_Blight",
    "Tomato_Healthy",
    "Maize_Healthy",
    "Cocoa_Healthy",
    "Cocoa_Pod_Borer",
    "Cocoa_Black_Pod_Rot",
    "Tomato_Leaf_Mold",
    "Maize_Common_Rust",
    "Tomato_Early_Blight"
]

#Loading model
class GreenPulseCustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(GreenPulseCustomCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)

        return x


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = GreenPulseCustomCNN(
        num_classes=len(CLASS_NAMES)
    ).to(device)

    model.load_state_dict(
        torch.load(
            "models/custom_greenpulse_model.pth",
            map_location=device
        )
    )
    model.eval()
    return model, device

#Image Preprocessing
inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

#Prediction Function
def predict_disease(image, model, device):
    image = image.convert("RGB")

    image_tensor = inference_transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    with torch.inference_mode():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_class.item()
    disease = CLASS_NAMES[predicted_index]
    confidence = confidence.item()

    return disease, confidence


col1, col2 = st.columns([5, 1])

with col1:
    st.markdown(
        '<div class="hero-title">🌿 GreenPulse</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="hero-subtitle">'
        'AI-powered crop disease detection for healthier crops.'
        '</div>',
        unsafe_allow_html=True
    )

with col2:
    language = st.selectbox(
        "🌍 Language",
        ["English", "Twi", "Ga", "Ewe"]
    )

st.divider()

left_column, right_column = st.columns(
    [1, 1],
    gap="large"
)

with left_column:

    st.markdown(
        '<div class="section-title">'
        '📷 Upload or Capture Image'
        '</div>',
        unsafe_allow_html=True
    )

    image_source = st.radio(
        "Choose Image Source",
        ["Upload Image", "Take Photo"],
        horizontal=True
    )

    image = None

    if image_source == "Upload Image":

        uploaded_file = st.file_uploader(
            "Upload a crop leaf image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:
            image = Image.open(uploaded_file)

    else:

        camera_photo = st.camera_input(
            "Take a picture of the leaf"
        )

        if camera_photo:
            image = Image.open(camera_photo)

    if image:

        st.image(
            image,
            caption="Leaf Preview",
            use_container_width=True
        )

        st.info(
            "💡 Tip: Use a clear photo with good lighting "
            "for the best prediction."
        )

        if st.button(
            "🔍 Detect Disease",
            type="primary",
            use_container_width=True):

            with st.spinner("Analyzing the leaf..."):

                model, device = load_model()

                disease, confidence = predict_disease(
                    image,
                    model,
                    device
                )

                st.session_state["prediction_made"] = True
                st.session_state["disease"] = disease
                st.session_state["confidence"] = confidence

            st.rerun()


with right_column:

    st.markdown(
        '<div class="section-title">'
        '🔬 Prediction Result'
        '</div>',
        unsafe_allow_html=True
    )

    if "prediction_made" in st.session_state:

        disease = st.session_state["disease"]
        confidence = st.session_state["confidence"]

        display_name = disease.replace("_", " ")

        confidence_percent = confidence * 100

        st.markdown(
            f"""<div class="prediction-card">
                    <div class="prediction-label">DETECTED CONDITION</div>
                    <div class="prediction-name">🌱 {display_name}</div>
                    <div class="confidence-label">CONFIDENCE</div>
                    <div class="confidence-value">{confidence_percent:.1f}%</div>
                </div>""",unsafe_allow_html=True)

        st.progress(confidence)

        st.markdown(
            """<div class="model-info">
            🤖 <b>GreenPulse AI</b><br>
                Prediction generated using the trained crop disease classification model.
                </div>""",unsafe_allow_html=True)

    else:
        st.markdown(
            """<div class="ready-card">
            <div class="ready-title">
            🌱 Ready to analyze
            </div>
            <div class="ready-text">
            Upload or capture a crop leaf image
            on the left, then click
            <b>Detect Disease</b> to let
            GreenPulse analyze it.
            </div>
            </div>""",unsafe_allow_html=True)