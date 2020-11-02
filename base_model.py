# PREPARE IMAGES DATA
# We will use the pre-trained model called VGG and available in keras to interpret the content of the photos.
# We extract the features of the photos and save them to file.

import os
import pickle
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Model


def extract_features(directory):
    print('Extracting features...')

    # Load model
    model = VGG16()
    # Re-structure the model (we remove the last layer from the model because we don't need to classify the photos)
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # Print the summary of the model
    print(model.summary())
    # Init features dictionary
    features = dict()

    for name in os.listdir(directory):
        # Load image from file
        filename = os.path.join(directory, name)
        image = load_img(filename, target_size=(224, 224))
        # Convert the image pixels to a numpy array
        image = img_to_array(image)
        # Reshape image for the model
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # Prepare image for the VGG model
        image = preprocess_input(image)
        # Extract features
        feature = model.predict(image)
        # Get image id
        image_id = name.split('.')[0]
        # Store feature
        features[image_id] = feature
        print('>%s' % name)

    return features


# Extract features from all images
features = extract_features('Flicker8k_Dataset')
print('Extracted Features: %d' % len(features))

# Save to file
pickle.dump(features, open('features.pkl', 'wb'))