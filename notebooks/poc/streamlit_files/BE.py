import streamlit as st
from PIL import Image
import numpy as np
from streamlit import caching
import pathlib
import os
import time
# from BEX_utils_animal import dummy_model
from pre_process_cropping_AP import BEX_cropping

from utils import *


def main():
    st.sidebar.image("bex_cube_certified.png", use_column_width=True)
    add_selectbox = st.sidebar.selectbox("Machine model operations",("Home screen","Evaluate image", "View Model data"))
    
    
    if add_selectbox == "Home screen":
        st.write("# Lets empower your brain!") 
        st.image('HS_brain_scan.jpg', width = 300)
 
# ---------------------------------------------------


    if add_selectbox == "Evaluate image":
        st.write("# Lets analyse your brain")
        filename = st.file_uploader(label = 'Drag and drop file (image of brain MRI) to examine')
        if add_selectbox == "Evaluate image":
            if st.button("Put model to work!"):
                get_path = os.getcwd()
                src_path = pathlib.Path(get_path)
                comp_path = src_path / filename.name        # issues with streamlit and path
                

                
                if filename is not None:
                    image = Image.open(filename)
                    img_array = np.array(image)
                    # st.write((np.stack(img_array,img_array, axis=2).shape))
                # pipelining for size and cropping
                # obj = BEX_cropping(comp_path)
                obj = BEX_cropping()
                st.write(filename.name)
                np_croppped = obj.calculate_cropping(img_array)
                #st.image(np_croppped)
                pred = model.predict(np_croppped)
                result_str = ""
                if pred < 0.5:
                    pct = (0.5 - pred[0][0]) * 2
                else:
                    pct = (pred[0][0] - 0.5)*2
                result_str = f"{CLASSES[int(pred+0.5)]} with {round(pct*100, 1)}% certainty"

                #st.write(f"pred output:{round(pred[0][0], 3)}")
                image, heatmap, superimposed_img = model.grad_cam(np_croppped)
                col11, col12, col13 = st.columns(3)
                with col11:
                    st.write("Original picture")
                    st.image(image)
                with col12:
                    st.write("With heatcam overlay")
                    st.image(superimposed_img)
                with col13:
                    st.write("Heatcam")
                    uint_heatmap = (np.uint8(255 * heatmap))
                    st.image(cv2.applyColorMap(uint_heatmap, cv2.COLORMAP_OCEAN))
                
                
                st.write(result_str)

# ---------------------------------------------------


    if add_selectbox == "View Model data":
        st.write("# Metrics and stuff")
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open("train_eval.png")     
            st.image(image, caption="Plots from training model")
        with col2:
            image = Image.open("train_eval2.png")     
            st.image(image, caption="Plots from training model")




if __name__ == "__main__":
    CLASSES = {0 : "Cancer found", 1 : "No cancer"}

    model = Model_Class('vgg19_MRI_Baseline_3.h5')
    main()