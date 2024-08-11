import streamlit as st
import numpy as np
import librosa
import pickle
import sounddevice as sd
from PIL import Image
import tensorflow as tf
import cv2
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, WebRtcMode, VideoHTMLAttributes

# Login credentials
LOGIN_USER = 'krotz'
LOGIN_PASSWORD = '123456'

# Load models
models = {}
machine_types = ['fan', 'pump', 'valve', 'slider']
for machine in machine_types:
    try:
        with open(f'{machine}_model.pkl', 'rb') as file:
            models[machine] = pickle.load(file)
    except FileNotFoundError:
        models[machine] = None

# Images for each machine type
images = {
    'slider': 'sll.gif',
    'fan': 'engine-plane.gif',
    'pump': 'puupp.gif',
    'valve': 'valll.gif'
}

# Function to preprocess and extract features for audio
def preprocess_audio_for_rf(audio, sr=22050, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return np.expand_dims(mfccs_mean, axis=0)

# Function to record audio from the microphone
def record_audio(duration=5, sr=22050):
    st.write("Recording...")
    recording = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    st.write("Recording completed.")
    return recording.flatten()

# Load image model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('my_modelk.keras')
    return model

image_model = load_model()

# Function to preprocess image for model prediction
def preprocess_image(image, target_size=(224, 224)):
    img = image.resize(target_size)
    img_array = np.array(img)
    
    if img_array.ndim == 2:  # If grayscale
        img_array = np.stack([img_array] * 3, axis=-1)  # Convert to RGB
    
    if img_array.shape[-1] != 3:
        raise ValueError("Image must have 3 channels (RGB).")
    
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Function to predict image class
def predict_image(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    return "Faulty" if prediction[0] < 0.5 else "Good"  # Assuming sigmoid output

# Video transformer for real-time camera input
class VideoTransformer(VideoTransformerBase):
    def __init__(self, model):
        self.model = model
        self.frame = None

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame = img
        return img

    def get_frame(self):
        return self.frame

# Streamlit login page
def login():
    st.markdown("""
        <style>
            .login-title {
                font-family: 'Bungee Spice', cursive;
                font-size: 36px;
                color: #D2691E;
                margin-bottom: 20px;
                text-align: center; /* Center-align text */
            }
            .login-container {
                background: black;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }
        </style>
        <div class='login-container'>
            <div class='login-title'> MAINTENANCE GUARDIAN</div>
        </div>
    """, unsafe_allow_html=True)

    st.image("ba.jpg", use_column_width=True)
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == LOGIN_USER and password == LOGIN_PASSWORD:
            st.session_state['logged_in'] = True
        else:
            st.error("Invalid username or password")
    
    st.markdown("</div></div>", unsafe_allow_html=True)

# Main app
def main():
    st.title("Maintenance Guardian")
    
    # Create a grid for two sections
    col1, col2 = st.columns(2)

    with col1:
        st.image("mmm.jpg", use_column_width=True)
        if st.button("Select Audio Anomaly Detection"):
            st.session_state['selected_section'] = 'audio'

    with col2:
        st.image("cam.png", use_column_width=True)
        if st.button("Select Image Anomaly Detection"):
            st.session_state['selected_section'] = 'image'
    
    if st.session_state.get('selected_section') == 'audio':
        st.header('Anomaly Detection for Slider, Fan, Pump, and Valve')
        st.write("Select a machine type and either record audio or upload an audio file to classify it as normal or abnormal.")
        
        machine_tabs = st.tabs(machine_types)
        for i, machine in enumerate(machine_types):
            with machine_tabs[i]:
                st.subheader(f"{machine.capitalize()} Anomaly Detection")
                
                if models[machine] is not None:
                    st.image(images[machine], caption=f"{machine.capitalize()} Machine")

                    option = st.selectbox(f"Choose an option for {machine.capitalize()}", ["Upload File", "Record Audio"], key=f"option_{machine}")

                    if option == "Upload File":
                        uploaded_file = st.file_uploader(f"Choose an audio file for {machine.capitalize()}", type=["wav"], key=f"upload_{machine}")

                        if uploaded_file is not None:
                            y, sr = librosa.load(uploaded_file, sr=22050)
                            features = preprocess_audio_for_rf(y)
                            prediction = models[machine].predict(features)
                            prediction_label = 'Abnormal' if prediction[0] == 1 else 'Normal'
                            st.write(f"The predicted label is: {prediction_label}")

                    elif option == "Record Audio":
                        duration = st.slider(f"Select recording duration for {machine.capitalize()} (seconds)", 1, 10, 5, key=f"duration_{machine}")
                        if st.button(f"Record {machine.capitalize()} Audio", key=f"record_{machine}"):
                            y = record_audio(duration=duration)
                            features = preprocess_audio_for_rf(y)
                            prediction = models[machine].predict(features)
                            prediction_label = 'Abnormal' if prediction[0] == 1 else 'Normal'
                            st.write(f"The predicted label is: {prediction_label}")

                else:
                    st.warning(f"Model for {machine} not found.")

    elif st.session_state.get('selected_section') == 'image':
        st.header("Machine Anomaly Detection with Image")

        option = st.selectbox("Choose input method", ("Upload Image", "Take Photo with Camera"))

        if option == "Upload Image":
            st.subheader("Upload an image of the machine state")
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                if st.button('Predict'):
                    result = predict_image(image, image_model)
                    st.write(f"Prediction: {result}")

        elif option == "Take Photo with Camera":
            st.subheader("Take a photo with your camera")
            webrtc_ctx = webrtc_streamer(
                key="example",
                mode=WebRtcMode.SENDRECV,
                video_transformer_factory=lambda: VideoTransformer(image_model),
                video_html_attrs=VideoHTMLAttributes(autoPlay=True, controls=True, style={"width": "100%"}),
            )

            if webrtc_ctx.video_transformer:
                if st.button('Capture and Predict'):
                    frame = webrtc_ctx.video_transformer.get_frame()
                    if frame is not None:
                        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        image = Image.fromarray(img_rgb)
                        st.image(image, caption='Captured Image', use_column_width=True)
                        
                        result = predict_image(image, image_model)
                        st.write(f"Prediction: {result}")
                    else:
                        st.write("No frame captured. Please try again.")

# Check if user is logged in
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    login()
else:
    main()
