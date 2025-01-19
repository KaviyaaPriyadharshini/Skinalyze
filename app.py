import os
import pandas as pd
import numpy as np
import cv2
import streamlit as st
from PIL import Image
from sklearn.metrics.pairwise import euclidean_distances
from tensorflow.keras.models import load_model

# Load dataset images and labels
def load_dataset_images(image_folder, labels_file):
    images = []
    labels = []
    labels_df = pd.read_csv(labels_file)  # Assuming labels file is a CSV with columns 'Image' and 'Severity Level'
    
    for index, row in labels_df.iterrows():
        img_path = os.path.join(image_folder, row['Image'])
        img = cv2.imread(img_path)
        img = cv2.resize(img, (128, 128))  # Resize images to a fixed size
        img = img / 255.0  # Normalize images to [0, 1]
        images.append(img)
        labels.append(row['Severity Level'])  # Severity levels are in the 'Severity Level' column
    
    return np.array(images), np.array(labels)

# Function to find the most similar image in the dataset
def find_closest_image(uploaded_image, dataset_images, dataset_labels):
    uploaded_image_flatten = uploaded_image.flatten().reshape(1, -1)
    dataset_images_flatten = dataset_images.reshape(dataset_images.shape[0], -1)
    
    # Calculate Euclidean distances
    distances = euclidean_distances(uploaded_image_flatten, dataset_images_flatten)
    
    # Find the index of the closest image
    closest_image_index = np.argmin(distances)
    
    # Return the label of the closest image
    return dataset_labels[closest_image_index]

# Preprocess uploaded image for comparison
def preprocess_image(image, img_size=(128, 128)):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)  # Convert to BGR format
    img = cv2.resize(img, img_size)  # Resize to match dataset image size
    img = img / 255.0  # Normalize to [0, 1]
    return img

# Load the skin type model
skin_type_model = load_model("/Users/daksha/Desktop/kavs/skin_type_model.keras")

# Function to predict skin type
def predict_skin_type(image):
    img = preprocess_image(image)
    img = img.reshape(1, 128, 128, 3)  # Reshape for the model
    prediction = skin_type_model.predict(img)
    class_labels = ['dry', 'normal', 'oily']
    predicted_class = class_labels[np.argmax(prediction)]
    return predicted_class.upper()  # Convert to uppercase for display

# Function to navigate between pages
def go_to_page(page_name):
    st.session_state["page"] = page_name

# Define CSS style
st.markdown(
    """
    <style>
    .stApp {
        background-color: #3C3C3C;
    }
    h1, h2, h3, h4, h5, h6, p, div, label {
        color: #FFFFFF; 
        font-size: 20px;  
    }
    .stTitle h1 {
        font-size: 60px !important;  
        font-weight: bold;  
        color: #FFFFFF;  
    }
    .stButton button {
        background-color: #00cc96;  
        color: white;
        border-radius: 15px;
        border: none;
        padding: 15px 30px;  
        font-size: 18px;  
        font-weight: bold;
    }
    .stFileUploader {
        color: #000000;  
        border-radius: 10px;  
        font-size: 16px;  
    }
    .stFileUploader .uploaded-file-text {
        color: #000000;  
    }
    .stButton button:hover {
        background-color: #00b386;  
    }
    
    .block-container .col {
        margin-right: 50px;  
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Home page
def show_home_page():
    st.title("Regular Skin Care Routine")  
    st.write("A guide to regular skin care with product recommendations and routines.")

    col1, col2 = st.columns([0.7, 0.5])  

    with col1:
        image = Image.open("/Users/daksha/Desktop/kavs/wallpaper.jpg")  
        st.image(image, caption='Regular Skin Care', use_column_width=True)

    with col2:
        st.write(""" 
        Maintaining a consistent skin care routine can make a huge difference in your skin's health.
        Make sure to stick to your routine daily and choose the right products for your skin type.
        """)
        if st.button("Get Started"):
            go_to_page("upload_page")  

# Severity levels and their mapped integer values
severity_mapping = {"Clear": 1, "Mild": 2, "Moderate": 3, "Severe": 4, "Very Severe": 5}

# Upload page with severity prediction and additional questions
def show_upload_page():
    st.title("Upload or Take a Picture")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Image uploaded successfully!")

        # Preprocess the uploaded image for severity prediction
        preprocessed_image = preprocess_image(image)

        # Load the dataset images and labels
        image_folder = "/Users/daksha/Desktop/kavs/JPEGImages"  # Path to your dataset images
        labels_file = "/Users/daksha/Desktop/kavs/acne_severity_results.csv"  # Path to your labels file
        dataset_images, dataset_labels = load_dataset_images(image_folder, labels_file)

        # Predict severity by finding the closest image in the dataset
        predicted_severity_label = find_closest_image(preprocessed_image, dataset_images, dataset_labels)
        
        # Display predicted acne severity
        st.markdown(f"<h3 style='color: white; font-size: 28px;'>Predicted Acne Severity:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: red; font-weight: bold; text-align: center;'>{predicted_severity_label}</h2>", unsafe_allow_html=True)

        # Use severity mapping to get the integer value for the slider
        severity_int_value = severity_mapping.get(predicted_severity_label, 1)
        st.slider("Predicted Acne Severity Level", min_value=1, max_value=5, value=severity_int_value, step=1, disabled=True)
        
        # Predict skin type
        predicted_skin_type = predict_skin_type(image)

        # Display predicted skin type
        st.markdown(f"<h3 style='color: white; font-size: 28px;'>Predicted Skin Type:</h3>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='color: blue; font-weight: bold; text-align: center;'>{predicted_skin_type}</h2>", unsafe_allow_html=True)
    if st.button("Go Back"):
        go_to_page("home_page")

# Page navigation logic
if "page" not in st.session_state:
    st.session_state["page"] = "home_page"

if st.session_state["page"] == "home_page":
    show_home_page()
elif st.session_state["page"] == "upload_page":
    show_upload_page()
