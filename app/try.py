import streamlit as st
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crop Disease Detector",
    page_icon="🌿",
    layout="wide"
)


# ============================================================
# CLASS NAMES
# ============================================================

CLASS_NAMES = [
    "Cocoa_Black_Pod_Rot",
    "Cocoa_Healthy",
    "Cocoa_Pod_Borer",
    "Maize_Common_Rust",
    "Maize_Healthy",
    "Maize_Northern_Leaf_Blight",
    "Tomato_Early_Blight",
    "Tomato_Healthy",
    "Tomato_Late_Blight",
    "Tomato_Leaf_Mold"
]


# ============================================================
# CUSTOM GREENPULSE CNN
# ============================================================

class GreenPulseCustomCNN(nn.Module):

    def __init__(self, num_classes=10):
        super(GreenPulseCustomCNN, self).__init__()

        # Convolutional Block 1
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm2d(32)

        # Convolutional Block 2
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm2d(64)

        # Convolutional Block 3
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm2d(128)

        # Pooling and dropout
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        self.dropout = nn.Dropout(0.3)

        # Fully connected layers
        self.fc1 = nn.Linear(
            128 * 28 * 28,
            256
        )

        self.fc2 = nn.Linear(
            256,
            num_classes
        )

    def forward(self, x):

        # Feature extraction
        x = self.pool(
            F.relu(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        x = self.pool(
            F.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        x = self.pool(
            F.relu(
                self.bn3(
                    self.conv3(x)
                )
            )
        )

        # Flatten
        x = torch.flatten(x, 1)

        # Classifier
        x = self.dropout(
            F.relu(
                self.fc1(x)
            )
        )

        x = self.fc2(x)

        return x


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "cpu"
    )

    model = GreenPulseCustomCNN(
        num_classes=len(CLASS_NAMES)
    )

    model.load_state_dict(
        torch.load(
            "models/custom_greenpulse_model.pth",
            map_location=device
        )
    )

    model = model.to(device)
    model.eval()

    return model, device


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

inference_transform = transforms.Compose([

    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_disease(image, model, device):

    # Convert image to RGB
    image = image.convert("RGB")

    # Apply same preprocessing used during testing
    image_tensor = inference_transform(image)

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    # Move to CPU/GPU
    image_tensor = image_tensor.to(device)

    # Prediction
    with torch.inference_mode():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted_class = torch.max(
            probabilities,
            dim=1
        )

    predicted_index = predicted_class.item()

    disease = CLASS_NAMES[predicted_index]

    confidence = confidence.item()

    return disease, confidence


# ============================================================
# HEADER
# ============================================================

col1, col2 = st.columns([5, 1])

with col1:

    st.title("🌿 Crop Disease Detector")

    st.caption(
        "Helping farmers identify crop diseases "
        "using Artificial Intelligence."
    )


with col2:

    language = st.selectbox(
        "🌍 Language",
        ["English", "Twi", "Ga", "Ewe"]
    )


st.divider()


# ============================================================
# MAIN COLUMNS
# ============================================================

left_column, right_column = st.columns([2, 3])


# ============================================================
# LEFT SIDE — IMAGE INPUT
# ============================================================

with left_column:

    st.subheader("📷 Upload or Capture Image")

    image_source = st.radio(
        "Choose Image Source",
        ["Upload Image", "Take Photo"]
    )

    image = None

    # Upload image
    if image_source == "Upload Image":

        uploaded_file = st.file_uploader(
            "Upload a crop leaf image",
            type=["jpg", "jpeg", "png"]
        )

        if uploaded_file:

            image = Image.open(uploaded_file)

    # Take photo
    else:

        camera_photo = st.camera_input(
            "Take a picture of the leaf"
        )

        if camera_photo:

            image = Image.open(camera_photo)


    # ========================================================
    # IMAGE PREVIEW
    # ========================================================

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


        # ====================================================
        # DETECT BUTTON
        # ====================================================

        if st.button(
            "🔍 Detect Disease",
            type="primary",
            use_container_width=True
        ):

            with st.spinner(
                "Analyzing the leaf..."
            ):

                model, device = load_model()

                disease, confidence = predict_disease(
                    image,
                    model,
                    device
                )


            # =================================================
            # RIGHT SIDE — RESULT
            # =================================================

            with right_column:

                st.subheader(
                    "Prediction Result"
                )

                display_name = disease.replace(
                    "_",
                    " "
                )

                st.success(
                    f"**Prediction:** {display_name}"
                )

                st.metric(
                    "Confidence",
                    f"{confidence * 100:.2f}%"
                )

                st.caption(
                    "Prediction generated by the "
                    "GreenPulse Custom CNN model."
                )