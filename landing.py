import streamlit as st
from streamlit.runtime.scriptrunner import get_script_run_ctx
import sys
import os
from PIL import Image
from st_aggrid import AgGrid
import pandas as pd
import altair as alt

# Set page configuration ONCE at the very beginning
st.set_page_config(
    page_title="Glaucoma Detection Hub",
    page_icon="👁️",
    layout="wide"
)

# Create a session state object if it doesn't exist
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'home'

# Navigation function
def navigate_to(page):
    st.session_state.current_page = page

#CSS Files

st.markdown("""
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,100..1000;1,9..40,100..1000&family=Fahkwang:ital,wght@0,200;0,300;0,400;0,500;0,600;0,700;1,200;1,300;1,400;1,500;1,600;1,700&family=Outfit:wght@100..900&display=swap');

            body * {
                font-family: "DM Sans", sans-serif !important; 
                font-weight: 500 !important;
                font-style: normal !important;    
            }

            [data-testid="stSidebar"] {
                padding-top: 0rem;
                background: #252c37; 
                color: #F8c617; 
                font-weight: bold;
                border: 2px solid #252c37;
                border-radius: 0px 30px 30px 0px; 
                box-shadow: 2px 2px 40px #F8c617;

            }

            .st-emotion-cache-1gulkj5{
                background-color: rgba(248, 198, 23, 0.21);
                border: 1px solid #F8c617;
                color: #252c37;
            }

             .st-au{
                background-color: rgba(248, 198, 23, 0.21);
                border: 1px solid #F8c617;
            }

            [data-testid="stSidebar"] .st-al{
                color: #F8c617;
            }
            .st-emotion-cache-1rci6ej {
                background-color: #252c37; 
                color: #F8c617; 
                font-weight: 900 !important;
                min-height: 4rem !important;
                border: none !important;
                justify-content: left;
            }
            .st-emotion-cache-12h5x7g p{
            font-family: "DM Sans", sans-serif !important;
                font-weight: 500 !important;
                font-size: 1.3rem !important;
            }
            
            [data-testid="stSidebar"] > div:first-child {
                padding-top: 0rem;
                margin-top: -1rem;
            }
            [data-testid="stSidebarNav"] {
                padding-top: 0rem;
            }

        .st-emotion-cache-1r6slb0{
            border: 2px solid #F8c617;
            padding: 0.8rem;
            border-radius: 20px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            }

        </style>
        """, unsafe_allow_html=True)

logo = Image.open('U-Net/logo.png')
# Sidebar navigation
with st.sidebar:
    st.button("🏠 Home", on_click=navigate_to, args=('home',), use_container_width=True)
    st.button("🔄 ResNet-18", on_click=navigate_to, args=('resnet',), use_container_width=True)
    st.button("🎯 YOLO & XGBoost", on_click=navigate_to, args=('yolo',), use_container_width=True)
    st.button("🔍 U-Net", on_click=navigate_to, args=('unet',), use_container_width=True)
    st.button("📊 Model Comparison", on_click=navigate_to, args=('comparison',), use_container_width=True)
    st.button("⚙️ Preprocessing", on_click=navigate_to, args=('preprocessing',), use_container_width=True)
    st.sidebar.info("""**Early Detection, Early Action: Your Vision, Our Priority** Empowering you with AI-driven insights, our tool leverages cutting-edge machine learning models to assist in the early detection of glaucoma.""")



profile = Image.open('U-Net/a.jpg')
def render_home():
    im1, im2 = st.columns([0.8, 0.2])
    with im1:
        st.markdown(""" <style> .main_title {
        font-size:3.1rem; color: #F8c617;} 
        </style> """, unsafe_allow_html=True)
        st.markdown('<h1 class="main_title"> GlaucoNetra - Glaucoma Detection Hub </h1>', unsafe_allow_html=True)
        st.write("Compare different deep learning models for Glaucoma detection")

    with im2:
        st.image(logo, width=200 )
        
    st.markdown("""
    ### Available Models
    Choose from our selection of advanced glaucoma detection models:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.info("### ResNet-18 Model")
        st.write("Uses ResNet-18 architecture for binary classification")
        st.button("Launch ResNet-18 Model", on_click=navigate_to, args=('resnet',))

    with col2:
        st.info("### YOLO & XGBoost Model")
        st.write("Combines YOLO object detection with XGBoost classification")
        st.button("Launch YOLO & XGBoost Model", on_click=navigate_to, args=('yolo',))

    with col3:
        st.info("### U-Net Model")
        st.write("Utilizes U-Net architecture for segmentation")
        st.button("Launch U-Net Model", on_click=navigate_to, args=('unet',))

    st.markdown("---")
    st.markdown(""" <style> .font {
        font-size:35px ; color: #F8c617;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font"> Glaucoma Diesease </p>', unsafe_allow_html=True)
        
    st.write('Glaucoma is a common cause of permanent blindness. It has globally affected 76 million individuals aged 40 to 80 by 2020, and the number is expected to rise to around 112 million due to the ageing population by the year 2040. However, many people do not aware of this situation as there are no early symptoms. The most prevalent chronic glaucoma condition arises when the trabecular meshwork of the eye becomes less effective in draining fluid. As this happens, eye pressure rises, leading to damage optic nerve. Thus, glaucoma care is essential for sustaining vision and quality of life, as it is a long-term neurodegenerative condition that can only control.')

    st.image(profile, width=700 )

    st.subheader("Causes for glaucoma")
    st.write('Ocular hypertension (increased pressure within the eye) is the most important risk factor for glaucoma, but only about 50% of people with primary open-angle glaucoma actually have elevated ocular pressure. Ocular hypertension—an intraocular pressure above the traditional threshold of 21 mmHg (2.8 kPa) or even above 24 mmHg (3.2 kPa)—is not necessarily a pathological condition, but it increases the risk of developing glaucoma. One study found a conversion rate of 18% within five years, meaning fewer than one in five people with elevated intraocular pressure will develop glaucomatous visual field loss over that period of time. It is a matter of debate whether every person with an elevated intraocular pressure should receive glaucoma therapy; currently, most ophthalmologists favor treatment of those with additional risk factors. Open-angle glaucoma accounts for 90% of glaucoma cases in the United States. Closed-angle glaucoma accounts for fewer than 10% of glaucoma cases in the United States, but as many as half of glaucoma cases in other nations (particularly East Asian countries).')

    st.subheader("Signs and symptoms")
    st.write("As open-angle glaucoma is usually painless with no symptoms early in the disease process, screening through regular eye exams is important. The only signs are gradually progressive visual field loss and optic nerve changes (increased cup-to-disc ratio on fundoscopic examination.")
    
    st.markdown(""" <style> .font {
        font-size:35px; color: #F8c617;} 
        </style> """, unsafe_allow_html=True)
    st.markdown('<p class="font"> Glaucoma Statistics </p>', unsafe_allow_html=True)

    @st.cache_data
    def data_upload():
        df= pd.read_csv("U-Net/Country Map  Estimates of Vision Loss1.csv")
        return df

    df=data_upload()
        #st.dataframe(data = df)

    st.header("The 10 countries with the highest number of persons with vision loss - 2022")
    st.write("As may be expected, these countries also have the largest populations. China and India together account for 49% of the world’s total burden of blindness and vision impairment, while their populations represent 37% of the global population.")
    AgGrid(df)

    @st.cache_data
    def data_upload1():
        df1= pd.read_csv("U-Net/Country Map  Estimates of Vision Loss2.csv")
        return df1

    df1=data_upload1()
        #st.dataframe(data = df)

    st.header("The 10 countries with the highest rates of vision loss - 2022")
    st.write("The comparative age-standardised prevalence rate can be helpful in providing a comparison of which country experiences the highest rates of vision impairment, regardless of age structure. India is the only country to appear on both ‘top 10’ lists, as it has the most vision impaired people, as well as the 5th highest overall rate of vision impairment.")
    AgGrid(df1)

    st.markdown(f'<h1 style="color:Red;font-size:25px;"> {"In 2022 in Sri Lanka, there were an estimated 3.9 million people with vision loss. Of these, 89,000 people were blind"}</h1>',unsafe_allow_html=True)
    energy_source = pd.DataFrame({
        "Types": ["Blindness","Mild","Mod-severe","Near","Blindness","Mild","Mod-severe","Near","Blindness","Mild","Mod-severe","Near","Blindness","Mild","Mod-severe","Near"],
        "Age prevalence %":  [0.60893,0.5401989,0.4371604,0.3701105,5.2044229,5.0064095,4.819523,4.6035189,5.4498435,5.6630457,5.5772188,5.1737878,5.7292227,5.5587928,5.3857774,5.2338178],
        "Year": ["1990","1990","1990","1990","2000","2000","2000","2000","2010","2010","2010","2010","2020","2020","2020","2020"]
        })
    
    bar_chart = alt.Chart(energy_source).mark_bar().encode(
            x="Year:O",
            y="Age prevalence %",
            color="Types:N"
        )
    st.altair_chart(bar_chart, use_container_width=True)
        
    st.markdown("""
    <style>
    /* Container for horizontal card layout */
    .card-container {
        display: flex;
        gap: 20px;
        justify-content: space-around;
        flex-wrap: wrap; /* Allows wrapping on smaller screens */
    }

    /* Individual card styling */
    .card {
        background-color: #f9f9f9;
        border-radius: 8px;
        border: 1px solid #252c37;
        box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
        padding: 15px;
        width: 30%;
        box-sizing: border-box;
    }
    
    /* Card title and text */
    .card h3 {
        color: #F8c617;
        margin-bottom: 10px;
    }
    .card p {
        color: #666;
        margin: 5px 0;
    }
    /* Responsive design */
    @media (max-width: 768px) {
        .card {
            width: 100%; /* Full-width on smaller screens */
        }
    }
    </style>
    <h2> Model Details </h2>
    <div class="card-container">
        <div class="card">
            <h3>ResNet-18 Model</h3>
            <p><strong>Architecture:</strong> Deep Residual Network (18 layers)</p>
            <p><strong>Task:</strong> Binary Classification (Normal/Glaucoma)</p>
            <p><strong>Accuracy:</strong> 99.83%</p>
        </div>
        <div class="card">
            <h3>YOLO & XGBoost Model</h3>
            <p><strong>Architecture:</strong> YOLO for segmentation + XGBoost for classification</p>
            <p><strong>Features:</strong> CDR, RDR, NRR metrics calculation</p>
            <p><strong>Details:</strong> Multiple detection metrics</p>
        </div>
        <div class="card">
            <h3>U-Net Model</h3>
            <p><strong>Architecture:</strong> U-Net for semantic segmentation</p>
            <p><strong>Specialization:</strong> Optic disc segmentation</p>
        </div>
    </div>
""", unsafe_allow_html=True)

def render_model(model_name):
    ima1, ima2 = st.columns([0.8, 0.2])
    with ima1:
        st.title(f"{model_name} Model")
        st.write(f"This is the {model_name} model interface.")
    with ima2:
        st.image(logo, width=150 )
    
    # Import and run the specific model code here
    try:
       if model_name == "ResNet-18":
           sys.path.append(os.path.join(os.getcwd(), "Resnet 18"))
           import resnet_glaucoma
           resnet_glaucoma.main()
       elif model_name == "YOLO & XGBoost":
           sys.path.append(os.path.join(os.getcwd(), "YOLO and XGBoost"))
           import yolov8_glaucoma_
           yolov8_glaucoma_.main()  # Call main function here
       elif model_name == "U-Net":
           sys.path.append(os.path.join(os.getcwd(), "U-Net"))
           import glaucocare
           glaucocare.glaucoma_detection_app()
    except Exception as e:
        st.error(f"Error loading {model_name} model: {str(e)}")
        st.write("Please make sure all required files and dependencies are available.")

# Main content router
if st.session_state.current_page == 'home':
    render_home()
elif st.session_state.current_page == 'comparison':
    # Directly import and call the function from comparison.py
    try:
        sys.path.append(os.path.join(os.getcwd(), "comparison"))
        import comp
        comp.main()  # Assuming the main function renders comparison content
    except Exception as e:
        st.error(f"Error loading comparison page: {str(e)}")
elif st.session_state.current_page == 'resnet':
    render_model("ResNet-18")
elif st.session_state.current_page == 'yolo':
    render_model("YOLO & XGBoost")
elif st.session_state.current_page == 'unet':
    render_model("U-Net")
elif st.session_state.current_page == 'preprocessing':
    try:
        import preprocessing_glaucoma
        preprocessing_glaucoma.main()
    except Exception as e:
        st.error(f"Error loading preprocessing tool: {str(e)}")
        st.write("Please make sure all required files and dependencies are available.")

# Add a home button in the footer for easy navigation
if st.session_state.current_page != 'home':
    st.markdown("---")
    st.button("🏠 Return to Home", on_click=navigate_to, args=('home',))
