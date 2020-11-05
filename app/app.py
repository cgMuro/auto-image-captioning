import os
import time
import base64
import streamlit as st
import numpy as np
from PIL import Image
import io
# Import functions to make the prediction
from predict import model, tokenizer, maxlength
from predict import extract_features, generate_description


# Call all the necessary function needed to generate the caption
def get_caption(img):
    # Extract features
    img = extract_features(img)
    # Get description
    description = generate_description(model, tokenizer, img, maxlength)
    # Remove start and end sequences
    description = ' '.join(description.split()[1:-1])

    return description



# Set title
st.title('Auto Image Captioning')

# Get image
input_image = st.file_uploader("Upload an image to caption", type=["png", "jpg", "jpeg"])

if input_image:
    # Display uploaded image
    input_image = Image.open(input_image)
    st.image(input_image, use_column_width=True)

    # Get description of the image
    input_image_resized = input_image.resize((224, 224))    # Resize image
    description = get_caption(input_image_resized)

    # Write caption
    st.write('The computer given description is...')
    st.write(f'**{description}**')
