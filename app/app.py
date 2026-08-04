import streamlit as st
from PIL import Image

st.set_page_config(
    page_title="Crop Disease Detector",
    page_icon="🌿",
    layout="wide"
)

col1, col2 = st.columns([5, 1])

with col1:
    st.title("🌿 Crop Disease Detector")
    st.caption("Helping farmers identify crop diseases using Artificial Intelligence.")

with col2:
    language = st.selectbox(
        "🌍 Language",
        ["English", "Twi", "Ga", "Ewe"]
    )
st.divider()

left_column,right_column = st.columns([2,3])

with left_column:
    st.subheader("📷 Upload or Capture Image")
    image_source = st.radio("Choose Image Source",
        ["Upload Image", "Take Photo"])

    image= None
    if image_source=="Upload Image":
        uploaded_file = st.file_uploader(
            "Upload a crop leaf image",
            type=["jpg","jpeg","png"]
        )
        if uploaded_file:
            image = Image.open(uploaded_file)

    else:
        camera_photo = st.camera_input("Take a picture of the leaf")
        if camera_photo:
            image = Image.open(camera_photo)

    if image:
        st.image(
            image,caption="Leaf Preview",
            use_container_width=True
        )
        st.info("💡 Tip: Use a clear photo with good lighting for the best prediction.")

        if st.button("🔍 Detect Disease"):
            st.success("Model will analyze the image here.")

