import streamlit as st
from pathlib import Path
from PIL import Image
from io import BytesIO
import os
import tensorflow as tf
import numpy as np
import requests
from keras.models import model_from_json

MODEL_PATH = 'model/saved_model'

st.set_option('deprecation.showfileUploaderEncoding', False)
st.title('Coffee Bean Classifier')
st.markdown("""
            ## How it Works
            - The model was for classifying 4 different roast types of coffee beans: green, light, medium and dark.
            - To use it, just copy the image link of a coffee bean you want.
            - just image files will work(jpg, png, jpeg).
            
            
            **:red[PROTIP:]** To increase sucess chance on classifying the right bean type, go for single coffee bean images!
        
            """)

st.text('Please provide a coffee bean image link like the placeholder below.')

@st.cache_data # for faster loading
def load_model(): 
    """Load a pretrained model. Must provive the destination folder for json and weight files."""
    # getting the pre-trained model                       
    model = tf.keras.models.load_model('xception_trained')
    return model
    
    
with st.spinner('Loading Model into memory....'):
    #model = load_model()
    #model = tf.keras.models.load_model('xception_trained')
    model  = load_model()
    #model = pickle.load(open('C:/Users/PICHAU/Desktop\FILES/studies/KAGGLE/e2e_xception_classification - Copia/model/saved_model/pickled_model.pkl' , 'rb'))
classes = ['dark', 'green', 'light', 'medium']


def scale_image(img):
  img = tf.cast(img, tf.float32)
  img /= 255

  return tf.image.resize(img,[224, 224])

def decode_img(image):
  img = tf.image.decode_jpeg(image, channels = 3)
  img = scale_image(img)
  return np.expand_dims(img, axis = 0)

def load_and_prep_image(filename, img_shape = 224):
  """Used to preprocess only local files"""
  img = tf.io.read_file(filename) #read image
  img = tf.image.decode_image(img) # decode the image to a tensor
  img = tf.image.resize(img, size = [img_shape, img_shape]) # resize the image
  img = img/255. # rescale the image
  return img

path = st.text_input('Placeholder image link:.', 'https://thumbs.dreamstime.com/b/one-green-coffee-bean-isolated-white-233230232.jpg')
# upload a file
if path is not None:
  content = requests.get(path).content

try:
  st.write('Predicted Class: ')
  with st.spinner('Classifying...'):
    label = np.argmax(model.predict(decode_img(content)), axis = 1)
    st.write(classes[label[0]])
  st.write("")
  image = Image.open(BytesIO(content))
  st.image(image, caption = 'Classyfying coffee bean image', use_column_width = True)
except ValueError:
  st.text('Invalid link format =(. Try choosing another image.')
