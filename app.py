from pathlib import Path
import PIL
import streamlit as st
import torch
import cv2
import settings
import helper
import os
from glob import glob

# Setting page layout
st.set_page_config(
    page_title="Egg Fertility Detection",  # Setting page title
    page_icon="🐣",     # Setting page icon
    layout="wide",      # Setting layout to wide
    initial_sidebar_state="expanded"    # Expanding sidebar by default
)

# Creating main page heading
st.title("Egg Fertility Detector")

st.sidebar.header("Egg Detector Configuration")

mlmodel_radio = st.sidebar.radio(
    "Task", ['Detection'])
conf = float(st.sidebar.slider("Select Model Confidence", 25, 100, 40)) / 100
if mlmodel_radio == 'Detection':
    dirpath_locator = settings.DETECT_LOCATOR
    model_path = Path(settings.DETECTION_MODEL)
try:
    model = helper.load_model(model_path)
except Exception as ex:
    print(ex)
    st.write(f"Unable to load model. Check the specified path: {model_path}")

source_img = None
st.sidebar.header("Image/Video Config")
source_radio = st.sidebar.radio(
    "Select Source", settings.SOURCES_LIST)

# body
# Initialize the counter
counter = 0
# If image is selected
if source_radio == settings.IMAGE:
    source_img = st.sidebar.file_uploader(
        "Choose an image...", type=("jpg", "jpeg", "png", 'bmp', 'webp'))
    save = False
    col1, col2 = st.columns(2)

    with col1:
        if source_img is None:
            default_image_path = str(settings.DEFAULT_IMAGE)
            image = PIL.Image.open(default_image_path)
            st.image(default_image_path, caption='Default Image',
                     use_column_width=True)
        else:
            image = PIL.Image.open(source_img)
            st.image(source_img, caption='Uploaded Image',
                     use_column_width=True)

    with col2:
        if source_img is None:
            default_detected_image_path = str(settings.DEFAULT_DETECT_IMAGE)
            image = PIL.Image.open(default_detected_image_path)
            st.image(default_detected_image_path, caption='Detected Image',
                     use_column_width=True)
        else:
            if st.sidebar.button('Detect Objects'):
                with torch.no_grad():
                    res = model.predict(image, save=save, save_txt=save, exist_ok=True, conf=conf)
                    boxes = res[0].boxes
                    res_plotted = res[0].plot()[:, :, ::-1]
                    st.image(res_plotted, caption='Detected Image', use_column_width=True)
                    
                    # Save the detected image
                    counter += 1  # Increment the counter by 1
                    output_folder = "results"
                    os.makedirs(output_folder, exist_ok=True)
                    image_name = f"detected_image_{counter}.png"
                    output_path = os.path.join(output_folder, image_name)
                    
                    # Save the original image using PIL or OpenCV
                    pil_image = PIL.Image.fromarray(res[0].orig_img)  # Convert numpy array to PIL Image
                    pil_image.save(output_path)  # Save the PIL Image
                    
                    IMAGE_DOWNLOAD_PATH = glob(os.path.join(output_folder, "detected_image_*.png"))
                    latest_image_path = max(IMAGE_DOWNLOAD_PATH, key=os.path.getctime)
                    with open(latest_image_path, 'rb') as fl:
                        st.download_button("Download latest object-detected image",data=fl,file_name=os.path.basename(latest_image_path),mime='image/png')
                        
                try:
                    with st.expander("Detection Results"):
                        for box in boxes:
                            st.write(box.xywh)
                except Exception as ex:
                    # st.write(ex)
                    st.write("No image is uploaded yet!")

elif source_radio == settings.WEBCAM:
    source_webcam = settings.WEBCAM_PATH
    if st.sidebar.button('Detect Objects'):
        vid_cap = cv2.VideoCapture(source_webcam)
        stframe = st.empty()
        counter = 0  # Initialize the counter
        while (vid_cap.isOpened()):
            success, image = vid_cap.read()
            if success:
                image = cv2.resize(image, (720, int(720*(9/16))))
                res = model.predict(image, conf=conf)
                res_plotted = res[0].plot()
                stframe.image(res_plotted,
                              caption='Detected Video',
                              channels="BGR",
                              use_column_width=True
                              )
                # Save the detected frame
                counter += 1  # Increment the counter by 1
                output_folder = "results"
                os.makedirs(output_folder, exist_ok=True)
                image_name = f"detected_frame_{counter}.jpg"  # Append the counter to the image name
                output_path = os.path.join(output_folder, image_name)
                cv2.imwrite(output_path, res_plotted)
