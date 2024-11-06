import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import torch.nn as nn
import os

def load_model():
    model = models.resnet18(pretrained=False)
    model.fc = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(512, 2))
    model.load_state_dict(torch.load(os.path.join('models', 'glaucoma_model.pth'), map_location=torch.device('cpu')))
    model.eval()
    return model

def main():
    st.markdown(""" <style> .main_title {
        font-size:3.1rem; color: #F8c617;} 
        img {
                border-radius: 20px; 
                box-shadow: 2px 2px 40px #dedede;
            }
        </style> """, unsafe_allow_html=True)
    st.markdown('<h1 class="main_title"> AI-Powered Glaucoma Detection using ResNet-18</h1>', unsafe_allow_html=True)
    st.header("Early Detection of Glaucoma from Retina Images")
    st.write("""
        This application uses a deep learning model to analyze retina images and detect signs of glaucoma. 
        Upload a fundus image, and the system will predict whether glaucoma is present or not. 
        Early diagnosis is crucial for managing and treating glaucoma, and this tool provides a quick, AI-based assessment.
    """)

    model = load_model()

    # Image transformations
    trans = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    st.write("## Upload a Retina Image")
    file = st.file_uploader("Choose a fundus image:", type=["jpg", "jpeg", "png"])
    if file is not None:
        img = Image.open(file)
        st.image(img, caption='Uploaded Image', width=400)

        # Process the image and make a prediction
        image = trans(img).unsqueeze(0)
        out = model(image)
        _, pred = torch.max(out, 1)
        pre = pred.item()
        st.markdown("""
        <style>
        
        .result-box {
            border-radius: 8px;
            color: white;
            padding: 15px;
            margin-top: 15px;
            font-size: 1.8rem;
            font-weight: bold;
            text-align: center;
        }
        .success-box {
            background-color: #4CAF50; /* Green background for success */
        }
        .warning-box {
            background-color: #F44336; /* Red background for warning */
        }
        </style>
        """, unsafe_allow_html=True)

        classes = ['No Glaucoma', 'Glaucoma']
        st.write(f"### Prediction: **{classes[pre]}**")
        if pre == 1:
            st.markdown(
            f'<div class="result-box warning-box">⚠️ Glaucoma Detected: Please consult an eye specialist for further evaluation.</div>',
            unsafe_allow_html=True
        )
        else:
            st.markdown(
            f'<div class="result-box success-box">✅ No Glaucoma Detected: The eye appears healthy.</div>',
            unsafe_allow_html=True
        )

        # Methodology description
        st.write("""
            ---
            ### Methodology:
            - **Image Preprocessing**: Uploaded retina images are resized and normalized to match the model input requirements.
            - **Deep Learning Model (ResNet18)**: A ResNet18 model fine-tuned for glaucoma detection predicts the presence or absence of glaucoma based on the input image.
            - **Results**: The prediction result indicates whether glaucoma signs are detected in the retina image with model accuracy as 99.83.
        """)

if __name__ == "__main__":
    main()
