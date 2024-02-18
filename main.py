import streamlit as st  
import webbrowser
from PIL import Image
import os
from ultralytics import YOLO 
from roboflow import Roboflow

task = st.sidebar.radio("", ["Home", "Object Detection", "Instance Segmentation", "Custom Object Detection"])
st.title(task)

# define the destination_folder where all uploaded files should be stored
destination_folder = 'C:/Users/Rahat/My_Work/ML_Project/YOLOv8_Object_Detector/final_streamlit_app/uploaded_files'

# Create the destination folder if it doesn't exist
os.makedirs(destination_folder, exist_ok=True) 


# get just filename without extension
def extract_filename(uploaded_filename):
    file_name, file_extension = os.path.splitext(uploaded_filename)
    return file_name


def HomeView():
    st.header("Project: Object Detection Using Deep Learning CNN")
    

def DetectionView():
    st.header("Detect Objects Using Pre-annotated Dataset")
    model = YOLO('yolov8n.pt')
    file = st.file_uploader("Upload file")
    if file is not None:
        filename = file.name 
        file_path = f'{destination_folder}/{filename}'
        with open(file_path, 'wb') as f:
            file_content = file.read()
            f.write(file_content)
            
        result = model(file_path, show=False, conf=0.5, save=True)
        result_path = os.path.join(result[0].save_dir, filename)

        if file.type == 'video/mp4':
            file_name = extract_filename(filename)
            video_url = os.path.join(result[0].save_dir, f"{file_name}.avi")
            webbrowser.open(video_url)
        else:
            webbrowser.open(result_path)

    

def SegmentationView():
    st.header("Detect & Segment Instances Using Pre-annotated Dataset")
    model = YOLO('yolov8n-seg.pt')
    file = st.file_uploader("Upload file")
    if file is not None:
        filename = file.name 
        file_path = f'{destination_folder}/{filename}'
        with open(file_path, 'wb') as f:
            file_content = file.read()
            f.write(file_content)

        result = model(file_path, show=False, conf=0.5, save=True)
        result_path = os.path.join(result[0].save_dir, filename)
        if file.type == 'video/mp4':
            file_name = extract_filename(filename)
            video_url = os.path.join(result[0].save_dir, f"{file_name}.avi")
            webbrowser.open(video_url)
        else:
            webbrowser.open(result_path)


def CustomView():
    st.header("Detect Objects Using Custom Annotated Dataset")
    rf = Roboflow(api_key="485V2UYUIK961W1Rbxkc")
    project = rf.workspace().project("tanjil_identifier")
    model = project.version(1).model
    file = st.file_uploader("Upload file")
    if file is not None:
        # filename = file.name 
        # file_path = f'{destination_folder}/{filename}'
        # with open(file_path, 'wb') as f:
        #     file_content = file.read()
        #     f.write(file_content)
        
        # prediction = model.predict(file_path, confidence=10, overlap=50).save("result2.avi")
        prediction = model.predict(None, confidence=10, overlap=50)
        print(prediction.json())
        # webbrowser.open("result2.avi")
        # if file.type == 'video/mp4':
        #     file_name = extract_filename(filename)
        #     video_url = os.path.join(result[0].save_dir, f"{file_name}.avi")
        #     webbrowser.open(video_url)


# ["Home", "Object Detection", "Instance Segmentation", "Custom Object Detection"]
if task == 'Home':
    HomeView() 
elif task == 'Object Detection':
    DetectionView() 
elif task == 'Instance Segmentation':
    SegmentationView() 
elif task == 'Custom Object Detection':
    CustomView() 