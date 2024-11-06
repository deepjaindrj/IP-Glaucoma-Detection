import streamlit as st
import cv2
import joblib
import imutils
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from yolov8_metricas_disco_escavacao import CDR, RDR, NRR, BVR, BVR2, CDRvh, excentricidade
from xgboost import XGBClassifier
import os

def crop_image_with_margin(image, box):
    width, height = image.size
    x1, y1, x2, y2 = box

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    radius_x = (x2 - x1) / 2
    radius_y = (y2 - y1) / 2

    margin_x = int(radius_x)
    margin_y = int(radius_y)

    x1_margin = max(0, int(center_x - radius_x - margin_x))
    y1_margin = max(0, int(center_y - radius_y - margin_y))
    x2_margin = min(width, int(center_x + radius_x + margin_x))
    y2_margin = min(height, int(center_y + radius_y + margin_y))

    return image.crop((x1_margin, y1_margin, x2_margin, y2_margin))

def contours_(x_min, y_min, x_max, y_max):
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2

    semi_axis_x = (x_max - x_min) / 2
    semi_axis_y = (y_max - y_min) / 2

    return center_x, center_y, semi_axis_x, semi_axis_y

def draw_contours(image, result_disc, result_cup):
    new_img = image.copy()
    altura, largura, _ = image.shape
    mask_disc = np.zeros((altura, largura), dtype=np.uint8)
    mask_cup = np.zeros((altura, largura), dtype=np.uint8)

    if len(result_disc.boxes) > 0:
        best_disc = max(result_disc.boxes, key=lambda x: x.conf[0])
        x_min, y_min, x_max, y_max = map(int, best_disc.xyxy[0].tolist())
        center_x, center_y, semi_axis_x, semi_axis_y = contours_(x_min, y_min, x_max, y_max)
        cv2.ellipse(image, (int(center_x), int(center_y)), (int(semi_axis_x), int(semi_axis_y)), 0, 0, 360, (0, 255, 0), 2)
        cv2.ellipse(mask_disc, (int(center_x), int(center_y)), (int(semi_axis_x), int(semi_axis_y)), 0, 0, 360, (255, 255, 255), -1)
        cnts_disc = cv2.findContours(mask_disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_disc = imutils.grab_contours(cnts_disc)
        cnt_disc = max(cnts_disc, key=cv2.contourArea)

    if len(result_cup.boxes) > 0:
        best_cup = max(result_cup.boxes, key=lambda x: x.conf[0])
        x_min2, y_min2, x_max2, y_max2 = map(int, best_cup.xyxy[0].tolist())
        center_x2, center_y2, semi_axis_x2, semi_axis_y2 = contours_(x_min2, y_min2, x_max2, y_max2)
        cv2.ellipse(image, (int(center_x2), int(center_y2)), (int(semi_axis_x2), int(semi_axis_y2)), 0, 0, 360, (0, 255, 0), 2)
        cv2.ellipse(mask_cup, (int(center_x2), int(center_y2)), (int(semi_axis_x2), int(semi_axis_y2)), 0, 0, 360, (255, 255, 255), -1)
        cnts_cup = cv2.findContours(mask_cup, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts_cup = imutils.grab_contours(cnts_cup)
        cnt_cup = min(cnts_cup, key=cv2.contourArea)

    cdr = CDR(cnt_disc, cnt_cup)
    cdrv, cdrh = CDRvh(cnt_disc, cnt_cup)
    rdr = RDR(cnt_disc, cnt_cup)
    nrr = NRR(cnt_disc, cnt_cup, new_img)
    return cdr, cdrv, cdrh, rdr, nrr

def process_image(image_bytes, yolo_model_disc, yolo_model_cup, loaded_model, confidence_threshold=0.01):
    image = Image.open(image_bytes)
    image_np = np.array(image)

    results_disc = yolo_model_disc(image, conf=confidence_threshold)
    results_cup = yolo_model_cup(image, conf=confidence_threshold)

    cdr, cdrv, cdrh, rdr, nrr = draw_contours(image_np, results_disc[0], results_cup[0])

    pred = loaded_model.predict([[cdr, cdrv, cdrh, rdr, nrr]])
    res = 'Normal' if pred == 0 and cdr < 0.35 else 'Glaucoma'

    return image_np, cdr, cdrv, cdrh, rdr, nrr, res

def main():
    st.markdown(""" <style> .main_title {
        font-size:3.1rem; color: #F8c617;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<h1 class="main_title"> Glaucoma Detection System Using XGBoost and YOLO </h1>', unsafe_allow_html=True)
    st.header("Methodology")

    st.markdown("### Image Preprocessing")
    st.write("Uploaded retina images are analyzed for black pixel content to identify focus areas. If high black pixel ratio is detected, unnecessary regions are cropped, retaining only diagnostically relevant parts.")

    st.markdown("### Optic Disc and Cup Segmentation")
    st.write("YOLO models (disc.pt and cup.pt) segment the optic disc and cup. Detected regions are highlighted, and contours are extracted for feature analysis.")

    st.markdown("### Feature Extraction")
    st.write(
        """
        Key metrics calculated from contours:
        - **CDR (Cup-to-Disc Ratio)**: Measures optic cup relative to disc.
        - **CDRv & CDRh**: Vertical and horizontal cup-to-disc ratios.
        - **RDR (Rim-to-Disc Ratio)**: Neuroretinal rim width.
        - **NRR (Neuroretinal Rim)**: Neuroretinal rim thickness.
        These metrics quantify structural changes in the retina, such as increased cup size and reduced rim thickness.
        """
    )

    st.markdown("### Classification using XGBoost Model")
    st.write("The pre-trained XGBoost model (model_xgb.pkl) classifies the retina as 'Normal' or 'Glaucoma' based on extracted metrics.")
    
    # Load models
    yolo_model_disc = YOLO(os.path.join('models', 'disc.pt'))
    yolo_model_cup = YOLO(os.path.join('models', 'cup.pt'))
    loaded_model = joblib.load(os.path.join('models', 'model_xgb.pkl'))

    image_bytes = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])
    
    if image_bytes is not None:
        output_image, cdr, cdrv, cdrh, rdr, nrr, res = process_image(image_bytes, yolo_model_disc, yolo_model_cup, loaded_model)
        st.markdown("### Display Results")
        st.write("Original image, segmented optic disc and cup, calculated metrics, and model prediction are displayed below.") 
        
        col1, col2, col3 = st.columns(3)
        col1.image(image_bytes, caption='Original image', width=200)
        col2.image(output_image, caption='Optic Disc and Cup Segmentation', width=200)
        col3.header("Metrics")

        # Define custom CSS for the styling
        st.markdown("""
            <style>
            .metric-name {
                font-weight: bold;
            }
            .metric-value {
                color: #333; /* Dark color for values */
            }
            .normal-result {
                background-color: #4CAF50; /* Green for normal */
                font-weight: bold;
                    border-radius: 8px;
                    color: white;
                    font-size: 1rem;
                    font-weight: bold;
                    text-align: center;
                    height: 20%;
            }
            .glaucoma-result {
                color: #F44336; /* Red for glaucoma */
                font-weight: bold;
                    border-radius: 8px;
                    color: white;
                    font-size: 1rem;
                    font-weight: bold;
                    text-align: center;
                    height: 20%;
            }
            .result-div{
                    
                    }
            </style>
        """, unsafe_allow_html=True)

        # Conditional styling for result
        result_class = 'normal-result' if res == 'Normal' else 'glaucoma-result'

        # Display the metrics with custom styling
        col3.markdown(f"""
            <p><span class="metric-name">CDR:</span> <span class="metric-value">{cdr}</span></p>
            <p><span class="metric-name">CDRv:</span> <span class="metric-value">{cdrv}</span></p>
            <p><span class="metric-name">CDRh:</span> <span class="metric-value">{cdrh}</span></p>
            <p><span class="metric-name">RDR:</span> <span class="metric-value">{rdr}</span></p>
            <p><span class="metric-name">NRR:</span> <span class="metric-value">{nrr}</span></p>
            <div class="{result_class}"><p>{res}</p></div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
