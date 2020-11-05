import streamlit as st
from PIL import Image
# Import functions to make the prediction
from predict import model, tokenizer, maxlength
from predict import extract_features, generate_description


# Call all the necessary function needed to generate the caption
def get_caption(img):
    # Extract features
    with st.spinner('Extracting features from image...'):
        img = extract_features(img)
    # Get description
    with st.spinner('Generating description...'):
        description = generate_description(model, tokenizer, img, maxlength)
    # Remove start and end sequences
    description = ' '.join(description.split()[1:-1])

    return description



# Set title
st.title('Auto Image Captioning')

# Add sidebar with explanation
st.sidebar.header('What is this and how does it work?')
st.sidebar.write('''
This is an application that uses a neural network to auto caption images.                            
You can upload an image and by clicking the button below it get a caption from the model.                           
*Please note* that since the dataset I used to train the model was relatively small (due to the lack of computationally resources), sometimes it can be quite wrong.        

If you want to know more about this project and the code behind it check it out on [GitHub](https://github.com/cgMuro/auto-image-captioning)
''')

# Get image
input_image = st.file_uploader("Upload an image to caption", type=["png", "jpg", "jpeg"])

if input_image:
    # Display uploaded image
    input_image = Image.open(input_image)
    st.image(input_image, use_column_width=True)

    if st.button('Get Caption!'):
        # Get description of the image
        input_image_resized = input_image.resize((224, 224))    # Resize image
        description = get_caption(input_image_resized)

        # Write caption
        st.write(f'**{description}**')
        # Draw celebratory balloons
        st.balloons()
