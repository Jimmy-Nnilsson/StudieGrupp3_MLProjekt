import streamlit as st
from PIL import Image
import numpy as np
from streamlit import caching
import pathlib
import os
import time
from BEX_utils_animal import dummy_model
from pre_process_cropping_AP import BEX_cropping



def main():
    st.sidebar.image("bex_cube_certified.png", use_column_width=True)
    add_selectbox = st.sidebar.selectbox("Machine model operations",("Home screen","Evaluate image", "View Model data"))
    
    
    if add_selectbox == "Home screen":
        st.write("# Lets empower your brain!") 
        st.image('HS_brain_scan.jpg', width = 300)
 
# ---------------------------------------------------


    if add_selectbox == "Evaluate image":
        st.write("# Lets classify the animal... or analyse your brain")
        filename = st.file_uploader(label='Drop it like its hot')
        if add_selectbox == "Evaluate image":
            if st.button("Put model to work!"):
                get_path = os.getcwd()
                src_path = pathlib.Path(get_path)
                comp_path = src_path / filename.name        # issues with streamlit and path

                # pipelining for size and cropping
                obj = BEX_cropping(comp_path)      
                st.write(filename.name)
                np_croppped = obj.calculate_cropping()    
                st.image(np_croppped, caption="Model says: " + dummy_model(filename.name))

# ---------------------------------------------------


    if add_selectbox == "View Model data":
        st.write("# Metrics and stuff")
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open("train_eval.png")     
            st.image(image, caption="Train and evaluate")
        with col2:
            image = Image.open("train_eval2.png")     
            st.image(image, caption="Train afasdf f e")




if __name__ == "__main__":
    main()