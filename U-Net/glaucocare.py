import os
import cv2
import keras
import imutils
import subprocess
import numpy as np
import pandas as pd
import altair as alt
import streamlit as st
import tensorflow as tf
from io import StringIO
from keras import losses
from st_aggrid import AgGrid
from matplotlib import pyplot
from keras import preprocessing
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from keras.models import Model
import plotly.graph_objects as go
from keras.layers import ELU, ReLU
from keras.models import load_model
from keras.preprocessing import image
from skimage.transform import resize
from streamlit_option_menu import option_menu
from keras.models import Model, load_model
from keras_preprocessing.image import load_img
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.applications import ResNet50
from gradcamplusplus import grad_cam, grad_cam_plus
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.preprocessing.image import img_to_array
from keras.layers import Input, MaxPooling2D, AveragePooling2D, average
from keras.layers import concatenate, Conv2D, Conv2DTranspose, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, UpSampling2D
from keras.layers import Convolution2D, ZeroPadding2D, Embedding, LSTM, concatenate, Lambda, Conv2DTranspose, Cropping2D
from custom_model import *
from CDR import *

def download_model(url, file_path):
    if not os.path.isfile(file_path):
        print(f"Downloading model from {url} to {file_path}...")
        result = subprocess.run([f'curl --output {file_path} "{url}"'], shell=True)
        if result.returncode == 0:
            print(f"Downloaded {file_path} successfully.")
        else:
            print(f"Failed to download {file_path}. Check the URL and try again.")

def load_model_safely(model_path):
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        return model
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

model = load_model_safely("U-Net/sep_5.h5")
model1 = load_model_safely("U-Net/OD_Segmentation.h5")
model2 = load_model_safely("U-Net/OC_Segmentation.h5")

def preprocess(img, req_size = (224,224)):
    image = Image.fromarray(img.astype('uint8'))
    image = image.resize(req_size)
    face_array = img_to_array(image)
    face_array = np.expand_dims(face_array, 0)
    return face_array

def preprocess_image(img_path, target_size=(224,224)):
    image = Image.fromarray(img_path.astype('uint8'))
    image = image.resize(target_size)
    img = img_to_array(image)
    #img = np.expand_dims(img, 0)
    img /= 255
    return img

def show_imgwithheat(img_path, heatmap, alpha=0.5, return_array=False):
    img = np.array(Image.fromarray(img_path))
    #img = cv2.imread(img_path)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = (heatmap*255).astype("uint8")
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = heatmap * alpha + img
    superimposed_img = np.clip(superimposed_img, 0, 255).astype("uint8")
    superimposed_img = cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB)
    return superimposed_img

def import_and_predict(image_data, model):
    image = ImageOps.fit(image_data, (224,224),Image.LANCZOS)
    #image = image.convert('RGB')
    image = np.asarray(image)
    #st.image(image, channels='RGB')
    image = (image.astype(np.float32) / 255.0)
    img_reshape = image[np.newaxis,...]
    prediction = model.predict(img_reshape)
    return prediction

def read_input(path):
    x = cv2.imread(path)
    x = cv2.resize(x, (256, 256))
    b, g, r = cv2.split(x)
    x = cv2.merge((r, r, r))
    return x.reshape(256, 256, 3)/255.

def read_gt(path):
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (256, 256))
    return x/255.

def load_image(image):
    image = read_image(image)
    return img_load


def glaucoma_detection_app():
    act = ReLU
    logo = Image.open('U-Net/logo.png')
    profile = Image.open('U-Net/a.jpg')
    st.markdown(""" <style> .font {
        font-size:35px; color: #F8c617;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font"> Fundus Image Analysis For Glaucoma </p>', unsafe_allow_html=True)

    st.write("To determine whether glaucomatous symptoms are present in an eye fundus image, please upload the image through the pane that can be found below. Depending on your network connection, it will take about 1~3 minutes to present the result on the screen.")
    
    st.write("This is a simple image classification web app to predict glaucoma through fundus image of eye")
    st.write("Sample Data: [Fundus Images](https://drive.google.com/drive/folders/1rKa5xtzw4_8Y53Om4e5LIlH6Jhp3hAT8?usp=sharing)")
    st.write("Check out the [User Manual](https://drive.google.com/file/d/1TLZ8P4K6jfjeVNb3qou9TeK0KlXJbUGA/view?usp=share_link)")
  
    label_dict={1:'Glaucoma', 0:'Normal'}

    file = st.file_uploader("Please upload an image(jpg/png/jpeg/bmp) file", type=["jpg", "png", "jpeg", "bmp"], key="main_file_uploader")

    if file is not None:
        file_details = {"filename": file.name, "filetype": file.type,"filesize": file.size}
        file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)

        col_a, col_b, col_c = st.columns(3)

        with col_a:
            st.subheader("Input image")
            imageI = Image.open(file)
            #st.image(imageI, width=250, channels="BGR",use_column_width=True)
        
            opencv_image = cv2.imdecode(file_bytes, 1)
            st.image(opencv_image, width=225,channels="BGR",use_column_width=False)
            opencv_image_processed = preprocess(opencv_image)
            
        with col_b:
            st.subheader("Grad-CAM")
            last_conv_layer= "conv5_block3_out" 
            img_path = np.array(Image.open(file))
            img = preprocess_image(img_path)
            heatmap = grad_cam(model, img,label_name = ['Glaucoma', 'Normal'],)
            output = show_imgwithheat(img_path, heatmap)
            output = imutils.resize(output, width=100)
            st.image(output,use_column_width=False, width=225)
            
        with col_c:
            st.subheader("Grad-CAM++")
            last_conv_layer= "conv5_block3_out"
            img_path = np.array(Image.open(file))
            img = preprocess_image(img_path)
            heatmap_plus = grad_cam_plus(model, img)
            output = show_imgwithheat(img_path, heatmap_plus)
            output = imutils.resize(output, width=100)
            st.image(output,use_column_width=False, width=225)
        
        col1_a, col1_b = st.columns(2)
        with col1_a:
            st.subheader("Segmented Optic Disc")
            contour_img = np.array(Image.open(file))
            img = cv2.resize(contour_img, (256, 256))
            b, g, r = cv2.split(img)
            img_r = cv2.merge((b, b, b))/255.
            #img_r1= cv2.resize(img_r, (224,224))
            #st.image(img)
            
            disc_model = get_unet(do=0.25, activation=act)
            disc_model.load_weights('U-Net/OD_Segmentation.h5')

            cup_model = get_unet1(do=0.2, activation=act)
            cup_model.load_weights('U-Net/OC_Segmentation.h5')

            disc_pred = disc_model.predict(np.array([img_r]))
            disc_pred = np.clip(disc_pred, 0, 1)
            pred_disc = (disc_pred[0, :, :, 0]>0.5).astype(int)
            pred_disc = 255 * pred_disc#.*(pred_disc - np.min(pred_disc))/(np.max(pred_disc)-np.min(pred_disc))
            cv2.imwrite('temp_disc.png', pred_disc)

            disc = cv2.imread('temp_disc.png', cv2.IMREAD_GRAYSCALE)
            st.image(pred_disc, width=225)

            masked = cv2.bitwise_and(img, img, mask = disc)
            #st.image(disc)
            #st.image(masked, width=225)
            #plt.show()
            mb, mg, mr = cv2.split(masked)
            masked = cv2.merge((mg, mg, mg)) #Morphological segmentation for defining optic disc from Green channel and optic cup from Red channel

        with col1_b: #cup segmentation
            st.subheader("Segmented Optic Cup")
            cup_pred = cup_model.predict(np.array([masked]))
            pred_cup = (cup_pred[0, :, :, 0]>0.5).astype(int)
            pred_cup = cv2.bilateralFilter(cup_pred[0, :, :, 0],10,40,20)
            pred_cup = (pred_cup > 0.5).astype(int)
            pred_cup = resize(pred_cup, (512, 512))
            pred_cup = 255.*(pred_cup - np.min(pred_cup))/(np.max(pred_cup)-np.min(pred_cup))
            cv2.imwrite('temp_cup.png', pred_cup)
            cup = cv2.imread('temp_cup.png', cv2.IMREAD_GRAYSCALE)
            st.image(pred_cup,width=225,clamp = True)

        disc = resize(disc, (512, 512))
        cv2.imwrite('temp_disc.png', disc)
        disc = cv2.imread('temp_disc.png', cv2.IMREAD_GRAYSCALE)
        (thresh, disc) = cv2.threshold(disc, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite('temp_disc.png', disc)
        (thresh, cup) = cv2.threshold(cup, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        cup_img = Image.open('temp_cup.png')
        disc_img = Image.open('temp_disc.png')  
        #os.remove('temp_cup.png')
        #os.remove('temp_disc.png')
        
        st.markdown("***")
        st.subheader('Cup-to-Disc Ratio (CDR)')
        
        fig = go.Figure(data=[go.Table(header=dict(values=['A Scores', 'B Scores']),
                        cells=dict(values=[[100, 90, 80, 90], [95, 85, 75, 95]]))
                            ])

        st.markdown(f'<h1 style="color:Gray;font-size:20px;">{"The normal cup-to-disc ratio is less than 0.4mm. A large cup-to-disc ratio may imply glaucoma."} </h1>', unsafe_allow_html=True)
        dias, cup_dias = cal(cup, disc, 'r')
        ddls, disc_dias, minrim, minang, minind = DDLS(cup_img, disc_img, 5)
        CDR = cup_dias[0]/disc_dias[0]
        st.markdown(f'<h1 style="color:Black;font-size:25px;">{"CDR : %.5f" % CDR}</h1>', unsafe_allow_html=True)
        
        st.markdown("***")
        prediction = import_and_predict(imageI, model)
        pred = ((prediction[0][0]))
        #print (pred)
        result=np.argmax(prediction,axis=1)[0]
        #print (result)
        accuracy=float(np.max(prediction,axis=1)[0])
        #print(accuracy)
        label=label_dict[result]
        #print(label)
        # print(pred,result,accuracy)
        # response = {'prediction': {'result': label,'accuracy': accuracy}}
        # print(response)

        Normal_prob = "{:.2%}".format(1-pred)
        Glaucoma_prob = "{:.2%}".format(pred)
        if(pred> 0.5):
            st.markdown(f'<h1 style="color:Red;font-size:35px;">{""" Glaucoma Eye"""}</h1>', unsafe_allow_html=True)
            #st.text("The area in the image that is highlighted is thought to be glaucomatous.")
        else:
            st.markdown(f'<h1 style="color:Blue;font-size:35px;">{"""Healthy Eye"""}</h1>', unsafe_allow_html=True)
            
        st.subheader('Prediction Probability')
        col1, col2 = st.columns(2)
        col1.metric("Glaucoma", Glaucoma_prob)
        col2.metric("Normal", Normal_prob)

        st.caption("**Note:This is a prototype tool for glaucoma diagnosis, using experimental deep learning techniques. It is recommended to consult a medical doctor for a proper diagnosis.")


glaucoma_detection_app()