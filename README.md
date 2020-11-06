# Auto Image Captioning

## Project Overview
* I used a CNN neural network to analyze photos and extract the features and a LSTM neural network to generate, given a photo, a caption automatically
* To extract the features from the photos I used a pre-trained model from keras called VGG16
* The LSTM neural network was used along with GloVe embeddings
* Finally I used Streamlit to develop and then deploy with docker on heroku an application that can be used to generate captions from new images

**NOTE**: the large files such as the datasets, the glove file, the models were removed because of their size, since they are too big for github.

## Data
From this [repository](https://github.com/jbrownlee/Datasets) I downloaded the datasets from both training and testing:
* Flicker8k_Dataset
* Flicker8k_text

I cleaned the descriptions and saved them into a file called `descriptions.txt`.


## Model
I used the pre-trained model VGG16 to extract the features from all the photos needed.
Then I used the `glove.6B.100d.txt` file, which contains vectors of 100 dimensions mapped several words (find out more [here](https://github.com/stanfordnlp/GloVe)) to create an embedding dictionary and matrix later used in the neural network.

This is the structure of the neural network I used:
* ```Photo Features Extractor``` --> the <em>pre-trained VGG16 model</em> (the one we used to pre-process the images features)
* ``Sequence Processor`` --> <em>word embedding layer</em> + <em>LSTM layer</em>
* ``Decoder`` --> it process the outputs of the **Photo Feature Extractor** and **Sequence Processor** and merges them together using a Dense layer to make a prediction

Here's the complete structure of the model:
```
Model: "functional_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
input_6 (InputLayer)            [(None, 37)]         0                                            
__________________________________________________________________________________________________
input_5 (InputLayer)            [(None, 4096)]       0                                            
__________________________________________________________________________________________________
embedding (Embedding)           (None, 37, 100)      759300      input_6[0][0]                    
__________________________________________________________________________________________________
dropout_2 (Dropout)             (None, 4096)         0           input_5[0][0]                    
__________________________________________________________________________________________________
dropout_3 (Dropout)             (None, 37, 100)      0           embedding[0][0]                  
__________________________________________________________________________________________________
dense_2 (Dense)                 (None, 256)          1048832     dropout_2[0][0]                  
__________________________________________________________________________________________________
lstm (LSTM)                     (None, 256)          365568      dropout_3[0][0]                  
__________________________________________________________________________________________________
add (Add)                       (None, 256)          0           dense_2[0][0]                    
                                                                 lstm[0][0]                       
__________________________________________________________________________________________________
dense_3 (Dense)                 (None, 256)          65792       add[0][0]                        
__________________________________________________________________________________________________
dense_4 (Dense)                 (None, 7593)         1951401     dense_3[0][0]                    
==================================================================================================
Total params: 4,190,893
Trainable params: 3,431,593
Non-trainable params: 759,300
__________________________________________________________________________________________________
None
```
Due to the lack of computationally resources I used [Google Colab](https://colab.research.google.com/) to train the model for 30 epochs using progressive overloading.

Finally I evaluated and tested the model using the [BLEU score](https://en.wikipedia.org/wiki/BLEU).


## Predictions
I then created another file where are loaded the pre-trained model saved and the tokenizer to generate captions from unseen images.                      
Since the dataset is quite small for this kind of task, a lot of captions fail to completely describe the image, while others are able to do it partially.                     

Finally I built some logic with OpenCV to open the image on a new window, with the generate caption written on the bottom.


## Deployment
I created a minimal app with the help of [Streamlit](https://www.streamlit.io/), in which the user can upload an image and get the caption from the model I trained.          
This application was then build with docker and deployed on heroku.            
You can find it [here](https://app-image-captioning.herokuapp.com/).


## Resources:
* https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/

* https://www.youtube.com/watch?v=NmoW_AYWkb4

* https://github.com/jeffheaton/t81_558_deep_learning/blob/master/t81_558_class_10_4_captioning.ipynb


## Packages
**Python version**: 3.8                                   
**Packages**:
```
pip install pandas
pip install numpy  
pip install matplotlib
pip install tensorflow
pip install opencv-python
pip install nltk
pip install streamlit
```
