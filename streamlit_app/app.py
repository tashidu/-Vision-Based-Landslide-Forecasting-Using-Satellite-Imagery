import streamlit as st


st.set_page_config(
    page_title="Landslide Forecasting",
    page_icon="🛰️",
    layout="wide"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🛰️ Landslide Forecasting")

st.write(
    "Vision-Based Landslide Detection Using Satellite Imagery"
)

st.divider()


# --------------------------------------------------
# Main sections
# --------------------------------------------------

upload_col, info_col = st.columns([2, 1])


with upload_col:
    st.subheader("Upload Satellite Image")

    uploaded_file = st.file_uploader(
        "Choose a satellite image",
        type=["jpg", "jpeg", "png"],
        help="Upload a JPG or PNG satellite image."
    )


with info_col:
    st.subheader("About")

    st.write(
        "This system analyzes satellite imagery and predicts "
        "whether the input image is associated with a landslide."
    )

if uploaded_file is not None:
    st.subheader("Image Preview")

    st.image(
        uploaded_file,
        caption=uploaded_file.name,
        use_container_width=True
    )