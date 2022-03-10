import streamlit as st
from PIL import Image
import numpy as np
from streamlit import caching
from pathlib import Path
import os
import time
import cv2

from utils import BEX_cropping
from utils import Model_Class
from utils import crcm
from utils import model_preprocess


def main():
    if 'page' not in st.session_state:
        st.session_state['page'] = 'Home screen'
    
    st.sidebar.image("./pic/bex_cube_certified.png", use_column_width=True)
    page_selector = st.sidebar.selectbox("Machine model operations",("Home screen","Evaluate image", "View Model data"))
    if st.sidebar.button("go"):
        st.session_state['page'] = page_selector

    if st.session_state['page'] == "Home screen":
        st.write("# Lets empower your brain!") 
        st.image('./pic/HS_brain_scan.jpg', width = 300)

# ---------------------------------------------------


    if st.session_state['page'] == "Evaluate image":
        st.write("# Lets analyse your brain")
        filename = st.file_uploader(label = 'Drag and drop file (image of brain MRI) to examine')
        if st.session_state['page'] == "Evaluate image":
            if st.button("Put model to work!"):
                # get_path = os.getcwd()
                # src_path = Path(get_path)
                # # comp_path = src_path / filename.name        # issues with streamlit and path


                if filename is not None:
                    image = Image.open(filename)
                    img_array = np.array(image)

                obj = BEX_cropping()
                st.write(filename.name)
                np_croppped = obj.calculate_cropping(img_array)
                pred = model.predict(np_croppped)

                image, heatmap, superimposed_img = model.grad_cam(np_croppped)
                col11, col12, col13 = st.columns(3)
                with col11:
                    st.write("Preprocessed picture")
                    st.image(image)
                with col12:
                    st.write("With heatcam overlay")
                    st.image(superimposed_img)
                with col13:
                    st.write("Heatcam")
                    uint_heatmap = (np.uint8(255 * heatmap))
                    st.image(cv2.applyColorMap(uint_heatmap, cv2.COLORMAP_OCEAN))

                pred_class = pred[0][np.argmax(pred[0])]
                # st.write(f"{pred[0][1]} {CLASSES[np.argmax(pred[0])]}")
                st.write(f"{round(float(pred[0][1]), 2)} sick")
                st.write(f"pred output:{(pred[0][1])}")

# ---------------------------------------------------


    if st.session_state['page'] == "View Model data":
        st.write("# Metrics and stuff")
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open("./pic/train_eval.png")     
            st.image(image, caption="Plots from training model")
        with col2:
            image = Image.open("./pic/train_eval2.png")     
            st.image(image, caption="Plots from training model")


if __name__ == "__main__":
    CLASSES = {1 : "Sick", 0 : "Well"}
    model_path = Path(os.getcwd() )/ "model/model.h5"
    
    model = Model_Class(model_path)
    main()