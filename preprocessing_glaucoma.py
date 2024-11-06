import streamlit as st
from PIL import Image
import numpy as np
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from io import BytesIO
from reportlab.lib.utils import ImageReader
import cv2
import matplotlib.pyplot as plt
from skimage import color, feature

def apply_clahe(image_array):
    lab = cv2.cvtColor(image_array, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return rgb_clahe

def convert_to_grayscale(image_array):
    grayscale_image = Image.fromarray(image_array).convert('L')
    return np.array(grayscale_image)

def contrast_stretching(image_array, min_out=0, max_out=255, min_in=None, max_in=None):
    if min_in is None:
        min_in = image_array.min()
    if max_in is None:
        max_in = image_array.max()
    stretched = (image_array - min_in) * ((max_out - min_out) / (max_in - min_in)) + min_out
    return stretched.astype(np.uint8)

def convert_to_binary(image_array, threshold=128):
    binary_image = (image_array > threshold) * 255
    return binary_image.astype(np.uint8)

def compute_saliency(image_array):
    if image_array.ndim == 2:
        image_array_rgb = np.repeat(image_array[:, :, np.newaxis], 3, axis=-1)
    else:
        image_array_rgb = image_array
    lab_image = color.rgb2lab(image_array_rgb)
    luminance = lab_image[:, :, 0]
    saliency_map = feature.canny(luminance)
    saliency_map = (saliency_map * 255).astype(np.uint8)
    return saliency_map

def pseudocolor_saliency(saliency_map, cmap=plt.get_cmap('hot')):
    normalized_saliency = saliency_map.astype(np.float32) / 255.0
    pseudocolored_saliency = (cmap(normalized_saliency) * 255).astype(np.uint8)
    return pseudocolored_saliency, saliency_map

def fourier_transform_processing(image_array):
    f_transform = np.fft.fft2(image_array)
    f_transform_shifted = np.fft.fftshift(f_transform)
    magnitude_spectrum = np.log(np.abs(f_transform_shifted) + 1)
    magnitude_spectrum = (magnitude_spectrum / magnitude_spectrum.max() * 255).astype(np.uint8)
    return magnitude_spectrum

def create_pdf_report(original_img, clahe_img, grayscale_img, contrast_img, binary_img, 
                     saliency_map, pseudocolored_saliency, magnitude_spectrum, mean_magnitude):
    pdf_buffer = BytesIO()
    c = canvas.Canvas(pdf_buffer, pagesize=letter)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(100, 750, "Image Processing Report")
    
    image_width = 250
    image_height = 250
    
    def img_to_bytes(img_array):
        img_bytes = BytesIO()
        Image.fromarray(img_array).save(img_bytes, format='PNG')
        img_bytes.seek(0)
        return img_bytes
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 650, "Original Image")
    c.drawImage(ImageReader(img_to_bytes(original_img)), 100, 400, width=image_width, height=image_height)
    
    c.drawString(100, 350, "CLAHE Enhanced")
    c.drawImage(ImageReader(img_to_bytes(clahe_img)), 100, 100, width=image_width, height=image_height)
    
    c.showPage()
    c.drawString(100, 750, "Grayscale Image")
    c.drawImage(ImageReader(img_to_bytes(grayscale_img)), 100, 500, width=image_width, height=image_height)
    
    c.drawString(100, 450, "Contrast Enhanced")
    c.drawImage(ImageReader(img_to_bytes(contrast_img)), 100, 200, width=image_width, height=image_height)
    
    c.showPage()
    c.drawString(100, 750, "Binary Image")
    c.drawImage(ImageReader(img_to_bytes(binary_img)), 100, 500, width=image_width, height=image_height)
    
    c.drawString(100, 450, "Saliency Map")
    c.drawImage(ImageReader(img_to_bytes(saliency_map)), 100, 200, width=image_width, height=image_height)
    
    c.showPage()
    c.drawString(100, 750, "Pseudocolored Saliency")
    c.drawImage(ImageReader(img_to_bytes(pseudocolored_saliency)), 100, 500, width=image_width, height=image_height)
    
    c.drawString(100, 450, "Fourier Spectrum")
    c.drawImage(ImageReader(img_to_bytes(magnitude_spectrum)), 100, 200, width=image_width, height=image_height)
    
    c.setFont("Helvetica", 12)
    c.drawString(100, 150, f"Mean Spectrum Value: {mean_magnitude:.2f}")
    
    c.save()
    pdf_buffer.seek(0)
    return pdf_buffer.getvalue()

def main():
    st.title('Image Preprocessing Tool')

    uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        clahe_enhanced = apply_clahe(image_array)
        grayscale_img = convert_to_grayscale(clahe_enhanced)
        contrast_enhanced = contrast_stretching(grayscale_img)
        binary_img = convert_to_binary(contrast_enhanced, threshold=128)
        saliency_map = compute_saliency(grayscale_img)
        pseudocolored_saliency, _ = pseudocolor_saliency(saliency_map)
        magnitude_spectrum = fourier_transform_processing(contrast_enhanced)
        mean_magnitude = np.mean(magnitude_spectrum)
        
        st.subheader("Original and Enhanced Images")
        st.markdown(""" <style> .st-emotion-cache-183lzff {
        font-size:1.1rem; color: #252c37;} 
            img {
                border-radius: 20px; 
                box-shadow: 2px 2px 40px #dedede;
            }
        </style> """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        
        with col1:
            st.text("Original Image")
            st.image(image)
            buf = BytesIO()
            image.save(buf, format="PNG")
            st.download_button("Download Original", data=buf.getvalue(), file_name="original.png", mime="image/png")
        
        with col2:
            st.text("CLAHE Enhanced")
            st.image(clahe_enhanced)
            buf = BytesIO()
            Image.fromarray(clahe_enhanced).save(buf, format="PNG")
            st.download_button("Download CLAHE Enhanced", data=buf.getvalue(), file_name="clahe.png", mime="image/png")
        
        st.subheader("Grayscale Processing")
        col3, col4 = st.columns(2)
        
        with col3:
            st.text("Grayscale Image")
            st.image(grayscale_img)
            buf = BytesIO()
            Image.fromarray(grayscale_img).save(buf, format="PNG")
            st.download_button("Download Grayscale", data=buf.getvalue(), file_name="grayscale.png", mime="image/png")
        
        with col4:
            st.text("Contrast Enhanced")
            st.image(contrast_enhanced)
            buf = BytesIO()
            Image.fromarray(contrast_enhanced).save(buf, format="PNG")
            st.download_button("Download Contrast Enhanced", data=buf.getvalue(), file_name="contrast.png", mime="image/png")
        
        st.subheader("Binary and Saliency Processing")
        col5, col6 = st.columns(2)
        
        with col5:
            st.text("Binary Image")
            st.image(binary_img)
            buf = BytesIO()
            Image.fromarray(binary_img).save(buf, format="PNG")
            st.download_button("Download Binary", data=buf.getvalue(), file_name="binary.png", mime="image/png")
        
        with col6:
            st.text("Saliency Map")
            st.image(saliency_map)
            buf = BytesIO()
            Image.fromarray(saliency_map).save(buf, format="PNG")
            st.download_button("Download Saliency Map", data=buf.getvalue(), file_name="saliency.png", mime="image/png")
        
        st.subheader("Fourier Transform")
        col7, col8 = st.columns(2)
        
        with col7:
            st.text("Pseudocolored Saliency")
            st.image(pseudocolored_saliency)
            buf = BytesIO()
            Image.fromarray(pseudocolored_saliency).save(buf, format="PNG")
            st.download_button("Download Pseudocolored", data=buf.getvalue(), file_name="pseudocolored.png", mime="image/png")
        
        with col8:
            st.text("Magnitude Spectrum")
            st.image(magnitude_spectrum)
            buf = BytesIO()
            Image.fromarray(magnitude_spectrum).save(buf, format="PNG")
            st.download_button("Download Spectrum", data=buf.getvalue(), file_name="spectrum.png", mime="image/png")
        
        st.subheader("Generate Report")
        pdf_report = create_pdf_report(image_array, clahe_enhanced, grayscale_img, contrast_enhanced, binary_img, 
                                       saliency_map, pseudocolored_saliency, magnitude_spectrum, mean_magnitude)
        st.download_button("Download PDF Report", data=pdf_report, file_name="Image_Report.pdf", mime="application/pdf")
