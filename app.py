import streamlit as st
import torch
from detect import detect
from PIL import Image
from io import *
import glob
from datetime import datetime
import os
import wget
import time

## CFG
cfg_model_path = "models/yourModel.pt" 

cfg_enable_url_download = True
if cfg_enable_url_download:
    url = "https://archive.org/download/yoloLSmart/yoloNSmartPS128.pt" #Configure this if you set cfg_enable_url_download to True
    cfg_model_path = f"models/{url.split('/')[-1:][0]}" #config model path from url name
## END OF CFG

# if cfg_enable_url_download:
#     urls = [ "https://archive.org/download/yoloXSmart/yoloXSmart.pt","https://archive.org/download/yoloLSmart/yoloNSmartPS128.pt", "https://archive.org/download/yoloLSmart/yoloMSmart.pt", "https://archive.org/download/yoloLSmart/yoloSSmartPS128.pt", "https://archive.org/download/yoloLSmart/yoloNSmartPS128.pt"] #Configure this if you set cfg_enable_url_download to True
#     cfg_model_path = f"models/{urls[0].split('/')[-1:][0]}"
#     i = 1
#     for i in range(len(urls)):
#         cfg_model_path += f" models/{urls[i].split('/')[-1:][0]}" #config model path from url name   
#          #cfg_model_path = f"models/{url.split('/')[-1:][0]}" #config model path from url name
# ## END OF CFG






def imageInput(device, src):
    
    if src == 'Upload your own data.':
        image_file = st.file_uploader("Upload An Image", type=['png', 'jpeg', 'jpg'])
        col1, col2 = st.columns(2)
        if image_file is not None:
            img = Image.open(image_file)
            with col1:
                st.image(img, caption='Uploaded Image', use_column_width='always')
            ts = datetime.timestamp(datetime.now())
            imgpath = os.path.join('data/uploads', str(ts)+image_file.name)
            outputpath = os.path.join('data/outputs', os.path.basename(imgpath))
            with open(imgpath, mode="wb") as f:
                f.write(image_file.getbuffer())

            #call Model prediction--
            model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True) 
            model.cuda() if device == 'cuda' else model.cpu()
            pred = model(imgpath)
            pred.render()  # render bbox in image
            for im in pred.ims:
                im_base64 = Image.fromarray(im)
                im_base64.save(outputpath)

            #--Display predicton
            
            img_ = Image.open(outputpath)
            with col2:
                st.image(img_, caption='Model Prediction(s)', use_column_width='always')

    elif src == 'From test set.': 
        # Image selector slider
        imgpath = glob.glob('data/images/*')
        imgsel = st.slider('Select random images from test set.', min_value=1, max_value=len(imgpath), step=1) 
        image_file = imgpath[imgsel-1]
        submit = st.button("Detect 🕵🏻!")
        col1, col2 = st.columns(2)
        with col1:
            img = Image.open(image_file)
            st.image(img, caption='Selected Image', use_column_width='always')
        with col2:            
            if image_file is not None and submit:
                #call Model prediction--
                model = torch.hub.load('ultralytics/yolov5', 'custom', path=cfg_model_path, force_reload=True) 
                pred = model(image_file)
                pred.render()  # render bbox in image
                for im in pred.ims:
                    im_base64 = Image.fromarray(im)
                    im_base64.save(os.path.join('data/outputs', os.path.basename(image_file)))
                #--Display predicton
                    img_ = Image.open(os.path.join('data/outputs', os.path.basename(image_file)))
                    st.image(img_, caption='Model Prediction(s)')




def videoInput(device, src):
    uploaded_video = st.file_uploader("Upload Video", type=['mp4', 'mpeg', 'mov'])
    if uploaded_video != None:

        ts = datetime.timestamp(datetime.now())
        imgpath = os.path.join('data/uploads', str(ts)+uploaded_video.name)
        outputpath = os.path.join('data/video_output', os.path.basename(imgpath))

        with open(imgpath, mode='wb') as f:
            f.write(uploaded_video.read())  # save video to disk

        st_video = open(imgpath, 'rb')
        video_bytes = st_video.read()
        st.video(video_bytes)
        st.write("Uploaded Video")
        detect(weights=cfg_model_path, source=imgpath, device=0) if device == 'cuda' else detect(weights=cfg_model_path, source=imgpath, device='cpu')
        st_video2 = open(outputpath, 'rb')
        video_bytes2 = st_video2.read()
        st.video(video_bytes2)
        st.write("Model Prediction")


def main():
    # -- Sidebar
    st.sidebar.title('⚙️Options')
    datasrc = st.sidebar.radio("Select input source.", ['Upload your own data.','From test set.'])
    
        
                
    option = st.sidebar.radio("Select input type.", ['Image', 'Video'])
   
    model_choice = st.sidebar.radio("Select Model.", ['yoloXSmart', 'yoloLSmart','yoloMSmart','yoloSSmartPS128','yoloNSmartPS128'])
    if torch.cuda.is_available():
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = False, index=1)
    else:
        deviceoption = st.sidebar.radio("Select compute Device.", ['cpu', 'cuda'], disabled = True, index=0)
    # -- End of Sidebar
    
    img_logo = Image.open(os.path.join('data/images', os.path.basename("smartathonLogo.png")))
    st.image(img_logo, caption='Smartathon')
    st.title("Visual Pollution Detection App")
    st.caption('developed by: Maseal Alghamdi, Abdullah Alshaya, Abdullah Alzaben, Anfal AlAwajy, Nada AlMugren, Sarah Alghamdi')
#     st.header('👀 Visual Pollution Detection')
    st.subheader('👈🏻 Select options left-haned menu bar.')
    
    if option == "Image":    
        imageInput(deviceoption, datasrc)
    elif option == "Video": 
        videoInput(deviceoption, datasrc)
        
    if model_choice == 'yoloXSmart':
        url = "https://archive.org/download/yoloXSmart/yoloXSmart.pt" #Configure this if you set cfg_enable_url_download to True
        cfg_model_path = f"models/{url.split('/')[-1:][0]}" #config model path from url name
        loadModel()
    
 
    elif model_choice == 'yoloLSmart':
        url = "https://archive.org/download/yoloLSmart/yoloLSmart.pt" #Configure this if you set cfg_enable_url_download to True
        cfg_model_path = f"models/{url.split('/')[-1:][0]}" #config model path from url name
        loadModel()
        
    elif model_choice == 'yoloMSmart':
        url = "https://archive.org/download/yoloLSmart/yoloMSmart.pt" #Configure this if you set cfg_enable_url_download to True
        cfg_model_path = f"models/{url.split('/')[-1:][0]}" #config model path from url name
        loadModel()
   
 
    elif model_choice == 'yoloSSmartPS128':
        url = "https://archive.org/download/yoloLSmart/yoloSSmartPS128.pt" #Configure this if you set cfg_enable_url_download to True
        cfg_model_path = f"models/{url.split('/')[-1:][0]}" #config model path from url name
        loadModel()
 
    else:
        url = "https://archive.org/download/yoloLSmart/yoloNSmartPS128.pt" #Configure this if you set cfg_enable_url_download to True
        cfg_model_path = f"models/{url.split('/')[-1:][0]}" #config model path from url name
        loadModel()
    

    

if __name__ == '__main__':
  
    main()

# Downlaod Model from url.    
@st.cache
def loadModel():
    start_dl = time.time()
    model_file = wget.download(url, out="models/")
    finished_dl = time.time()
    print(f"Model Downloaded, ETA:{finished_dl-start_dl}")
    
#     for url in urls:
#         start_dl = time.time()
#         model_file = wget.download(url, out="models/")
#         finished_dl = time.time()
#         print(f"Model Downloaded, ETA:{finished_dl-start_dl}")
    
if cfg_enable_url_download:
    loadModel()
