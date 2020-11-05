import pickle
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Load model
model = tf.keras.models.load_model('./app/model.h5')

# Load tokenizer
tokenizer = pickle.load(open('./app/tokenizer.pkl', 'rb'))

# Define maxlength
maxlength = 37


# Process image function
def extract_features(img):
    # Load model
    model = VGG16()
    # Re-structure the model (we remove the last layer from the model because we don't need to classify the photos)
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # Convert the image pixels to a numpy array
    image = img_to_array(img)
    # Reshape image for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Prepare image for the VGG model
    image = preprocess_input(image)
    # Extract features
    features = model.predict(image, verbose=0)

    return features


# GET CAPTION FUNCTIONS

# Function that given an integer returns the corresponding word based on the tokenization
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word

    return None


def generate_description(model, tokenizer, photo, maxlength):
    # Word that starts the sequence
    in_text = 'STARTSEQ'
    # Iterate over the sequence
    for _ in range(maxlength):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=maxlength)
        # Predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        # Get the integer with the biggest probability
        yhat = np.argmax(yhat)
        # Get the word corresponding to the integer the model returned
        word = word_for_id(yhat, tokenizer)
        # Stop if no valid word was predicted
        if word is None:
            break
        # Append the predicted word as input for the next word
        in_text += ' ' + word
        # Stop if we predicted the end of the sequence
        if word.upper() == 'ENDSEQ':
            break

    return in_text