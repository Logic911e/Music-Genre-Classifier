import streamlit as st
import tensorflow as tf
import numpy as np
import librosa
from tensorflow.image import resize
import os
os.makedirs("Test_Music", exist_ok=True)

#store model in cache
st.cache_resource()
#Load the trained model
def load_model():
    try:
        model = tf.keras.models.load_model('Trained_model.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
    
#Load and preprocess audio file
def load_preprocess_file(file_path,target_shape =(150,150)):
   data = [] 
   audio_data,sample_rate = librosa.load(file_path,sr = None)
   chunk_time = 6
   overlap_time = 3
   #Convert Duration to Sample
   chunk_sample = chunk_time*sample_rate
   overlap_samples = overlap_time * sample_rate
   total_chunk_num = int(np.ceil((len(audio_data)-chunk_sample)/(chunk_sample-overlap_samples)))+1
   #iterate over each chunk
   for i in range(total_chunk_num):
    #Calculate start and end indices of the chunk
    start = i*(chunk_sample-overlap_samples)
    end = start+chunk_sample
    #extract chunk audio
    chunk = audio_data[start:end]

    mel_spectrogram = librosa.feature.melspectrogram(y=chunk, sr= sample_rate)

    #Resize matrix based on provided target shape.
    mel_spectrogram = resize(np.expand_dims(mel_spectrogram,axis=-1),target_shape)
    #Append data to list
    data.append(mel_spectrogram)
   return np.array(data)

# predict the genre of the audio file
def model_prediction(X_test):
    model = load_model()
    y_pred = model.predict(X_test)
    predicted_categories = np.argmax(y_pred,axis=1)
    unique_elements,counts = np.unique(predicted_categories,return_counts=True)
    max_count = np.max(counts)
    max_elements = unique_elements[counts==max_count]
    return max_elements[0]



#Streamlit UI
st.sidebar.title("Dashboard")

app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Predict"])

# Home Page
if app_mode == "Home":
   st.markdown(
      """
      <style>
      .stApp {
            background-color: #000000;
            color: #ffffff;
            font-family: 'Arial', sans-serif;
        }
        h2, h3,h1, p, li, div  {
            color: #ffffff;
        }
        </style>
        """,
        unsafe_allow_html=True

      
   )
   st.markdown('''
   ## Welcome to my app 🙂''')
   image_path = '/Users/anjola/Downloads/Black and White Decorative Monoline Jazz Music YouTube Banner.png'
   st.image(image_path,use_container_width=True)

   st.markdown("""
    ### How it works:
    This app classifies music genres using a trained model. Upload an audio file, and the app will predict its genre.

    ### Supported Genres:
     - Pop
     - Rock
     - Hip-Hop
     - Jazz
     - Classical
     - Metal
     - Country
     - Reggae
     - Blues
     - Disco
               
    
    ### Get started
    Click on the **Predict** tab in the sidebar to upload an audio file and get predictions.         

    """)
elif app_mode == "About":
   st.markdown(
       """
       ### About the Project
       This project is a Music Genre Classification app built with Streamlit and TensorFlow. It allows users to upload audio files and predicts their genres using a pre-trained deep learning model.

       ### Model Details
       The model is trained on a diverse dataset of music tracks across various genres. It uses Mel spectrograms as input features and a convolutional neural network (CNN) architecture for classification.

       ### Acknowledgments
       - [Librosa](https://librosa.org/) for audio processing
       - [TensorFlow](https://www.tensorflow.org/) for deep learning
       - [Streamlit](https://streamlit.io/) for building the web app
       - [SpotlessTech](https://www.youtube.com/@SPOTLESSTECH) for the tutorial on building this app
       

       """
   )


elif app_mode == "Predict":
   st.header("Model Predictions")
   test_sound_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])
   if test_sound_file is not None:
      filepath ="Test_Music/"+test_sound_file.name
      with open(filepath, "wb") as f:
          f.write(test_sound_file.getbuffer())

    #play audio button
   if st.button("Play Audio"):
      st.audio(test_sound_file, format="audio/wav")

    #Predict button
   if(st.button("Predict")):
      with st.spinner("Will be done in two ticks.."):
        X_test = load_preprocess_file(filepath)
        res_index = model_prediction(X_test)
        genres = ['blues','classical','country','disco','hiphop','jazz','metal','pop','reggae','rock']

        st.markdown(
            "Model Prediction: It's a <span style='color: purple; font-weight: bold'>{}</span> song!".format(genres[res_index]),
            unsafe_allow_html=True )
        # st.markdown("Model Prediction: It's a :purple **{}** song!".format(genres[res_index])) 
        

