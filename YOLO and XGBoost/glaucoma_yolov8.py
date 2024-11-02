import streamlit as st
import cv2
import joblib
import imutils
import numpy as np
from PIL import Image
from io import BytesIO
from ultralytics import YOLO
from metricas_disco_escavacao import CDR, RDR, NRR, BVR, BVR2 , CDRvh, excentricidade
from xgboost import XGBClassifier

st.title("Glaucoma Detection System Using XGBoost and YOLO")
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

st.markdown("### Display Results")
st.write("Original image, segmented optic disc and cup, calculated metrics, and model prediction are displayed below.")
def crop_image_with_margin(image, box):
    width, height = image.size
    x1, y1, x2, y2 = box

    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    radius_x = (x2 - x1) / 2
    radius_y = (y2 - y1) / 2

    margin_x = int(radius_x )
    margin_y = int(radius_y )

    x1_margin = max(0, int(center_x - radius_x - margin_x))
    y1_margin = max(0, int(center_y - radius_y - margin_y))
    x2_margin = min(width, int(center_x + radius_x + margin_x))
    y2_margin = min(height, int(center_y + radius_y + margin_y))

    x1_crop = max(0, x1_margin)
    y1_crop = max(0, y1_margin)
    x2_crop = min(width, x2_margin)
    y2_crop = min(height, y2_margin)

    return image.crop((x1_crop, y1_crop, x2_crop, y2_crop))


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

def pil_image_to_bytes(image):
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def img_retina(image):

    image_ = Image.open(image)
    image_rgb = np.array(image_)

    black_pixels = np.sum(np.all(image_rgb  <= [20, 20, 20], axis=2))
    total_pixels = image_rgb.shape[0] * image_rgb.shape[1]
    black_pixel_ratio = black_pixels / total_pixels
    print(black_pixel_ratio)

    if black_pixel_ratio > 0.05:

        results_cup = yolo_model_cup(image_, conf=0.01)
        if results_cup and len(results_cup[0].boxes.xyxy) > 0:
            max_confidence = 0
            best_box = None
            for box, conf in zip(results_cup[0].boxes.xyxy, results_cup[0].boxes.conf):
                if conf > max_confidence:
                    max_confidence = conf
                    best_box = box

            if best_box is not None:
                box = best_box.cpu().numpy().astype(int).tolist()
                
                cropped_image = crop_image_with_margin(image_, box)
                
                image_final = pil_image_to_bytes(cropped_image)
    else:
        image_final = image

    return image_final
def process_image(image_bytes, yolo_model_disc, yolo_model_cup, confidence_threshold=0.01):
    
    path_input = img_retina(image_bytes)
    image = Image.open(path_input)
    image_np = np.array(image)

    results_disc = yolo_model_disc(image, conf=confidence_threshold)
    results_cup = yolo_model_cup(image, conf=confidence_threshold)

    cdr, cdrv, cdrh, rdr, nrr = draw_contours(image_np, results_disc[0], results_cup[0])

    pred = loaded_model.predict([[cdr, cdrv, cdrh, rdr,nrr]])
    if pred == 0 and cdr < 0.35:
        res = 'Normal'
    elif pred==1:
        res = 'Glaucoma'
    else:
        res = 'Glaucoma'

    return image_np, cdr, cdrv, cdrh, rdr,nrr, res

yolo_model_disc = YOLO('models/disc.pt')
yolo_model_cup = YOLO('models/cup.pt')
loaded_model = joblib.load('models/model_xgb.pkl')

#st.title("Detect Glaucoma")
image_bytes = st.file_uploader("Select an image", type=["jpg", "jpeg", "png"])

if image_bytes is not None:
    output_image, cdr, cdrv, cdrh, rdr,nrr, res = process_image(image_bytes, yolo_model_disc, yolo_model_cup)

    col1, col2, col3 = st.columns(3)

    col1.image(image_bytes, caption='Original image', width=200)
    col2.image(output_image, caption='Optic Disc and Cup Segmentation', width=200)
    col3.header("Metrics")
    if cdr is not None:
        col3.write(f"CDR: {cdr}")
    if cdrv is not None:
        col3.write(f"CDRv: {cdrv}")
    if cdrh is not None:
        col3.write(f"CDRh: {cdrh}")
    if rdr is not None:
        col3.write(f"RDR: {rdr}")
    if nrr is not None:
        col3.write(f"NRR: {nrr}")
    if res is not None:
        col3.write(f"Result: {res}")