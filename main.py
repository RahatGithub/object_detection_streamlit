import streamlit as st  
import webbrowser
import os
from PIL import Image
from ultralytics import YOLO 

# get just filename without extension
def extract_filename(uploaded_filename):
    file_name, file_extension = os.path.splitext(uploaded_filename)
    return file_name

# load the model
model = YOLO('yolov8n.pt')

# Destination folder for storing uploaded images
destination_folder = 'C:/Users/Rahat/My_Work/ML_Project/YOLOv8_Object_Detector/final_streamlit_app/uploaded_files'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True) 

file = st.file_uploader("Upload file")

if file is not None:
    filename = file.name 
    file_path = f'{destination_folder}/{filename}'
    with open(file_path, 'wb') as f:
        file_content = file.read()
        f.write(file_content)
    
    processed_file = model(file_path, show=False, conf=0.5, save=True)
    save_dir = 'C:/Users/Rahat/My_Work/ML_Project/YOLOv8_Object_Detector/final_streamlit_app/result_files'
    result_img_path = f'{save_dir}/{filename}'
    processed_file[0].save(result_img_path)
    image_to_show = Image.open(result_img_path)
    image_to_show.show()

    # st.write(file_path)
    # st.write(result[0].save_dir)
    # result_path = os.path.join(result[0].save_dir, filename)
    # st.write(result_path)
    # if file.type == 'video/mp4':
    #     file_name = extract_filename(filename)
    #     video_url = os.path.join(result[0].save_dir, f"{file_name}.avi")
    #     webbrowser.open(video_url)