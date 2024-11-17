import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import numpy as np
import cv2
import requests
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from io import BytesIO

# Initialize session state
if "page" not in st.session_state:
    st.session_state.page = "Home"  

# Set page configuration
st.set_page_config(page_title="EmotionFusion", layout="wide")

# Streamlit app interface
st.title("EmotionFusion – Multimodal Understanding of Human Emotions")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select an option",
                        ["Home", "Image Annotation", "Video Annotation",
                         "Live Cam"], key='navigation')

# Main content based on selected page

# Home page content
if page == "Home":
    # Title and introductory tagline
    st.title("Welcome to EmotionFusion")
    st.subheader("Discover Emotions, Uncover Insights")
    st.write(
        """
        **EmotionFusion** is your all-in-one platform that helps you analyze emotions from text, images, and live video feeds. Whether you're curious about your emotional state or want to explore how emotions are expressed, EmotionFusion offers instant, real-time analysis to give you valuable insights.
        """
    )

    # Key features section
    st.header("Key Features")
    st.write(
        """
        **1. Text Emotion Analysis**
        Type or paste any text to see what emotions are conveyed. Our tool breaks down the emotions hidden in your words.

        **2. Image & Video Emotion Detection**
        Upload photos or videos to analyze facial expressions and see the emotions behind them.

        **3. Real-Time Emotion Recognition**
        Turn on your webcam to get live emotion feedback as you interact. See how your facial expressions shift in real time!
        """
    )


    # How it works section
    st.header("How It Works")
    st.write(
        """
        1. **Input**: Choose to input text, upload an image or video, or use your live camera feed.
        2. **Emotion Detection**: Our system will analyze your input and detect emotions instantly.
        3. **Visualize**: See the results in easy-to-read charts and visualizations.
        4. **Download**: Save the analysis or share it for further exploration.
        """
    )

    # Why choose us section
    st.header("Why Choose EmotionFusion?")
    st.write(
        """
        - **Multiple Ways to Detect Emotions**: From text to images and live video, EmotionFusion offers diverse ways to analyze emotions.
        - **Real-Time Feedback**: Experience instant emotion analysis with our easy-to-use, live webcam feature.
        - **Simple and Intuitive**: Whether you're a researcher or just curious, our platform is designed for anyone to use without needing technical knowledge.
        """
    )
    
# Load the emotion recognition model
model_path = r"New_model.h5"
model = load_model(model_path)

# Load the MTCNN detector for face detection
detector = MTCNN()

# Define emotion labels based on the model's output
emotion_labels = [
    'Angry', 'Contempt', 'Disgust', 'Fear',
    'Happy', 'Neutral', 'Sad', 'Surprise'
]

# Scaling factors for emotions
emotion_scaling = {
    'angry': 1.0,
    'contempt': 1.0,
    'disgust': 1.0,
    'fear': 1.0,
    'happy': 1.0,
    'neutral': 1.0,
    'sad': 1.0,
    'surprise': 1.0,
}

# Preprocess face region for prediction
def preprocess_face(roi):
    roi = cv2.resize(roi, (112, 112))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi = roi.astype('float32') / 255.0
    roi = np.expand_dims(roi, axis=-1)
    roi = np.expand_dims(roi, axis=0)
    return roi

# Predict emotion and apply scaling
def predict_emotion(roi):
    preprocessed = preprocess_face(roi)
    predictions = model.predict(preprocessed)[0]

    # Apply scaling factors
    scaled_predictions = np.array([pred * emotion_scaling[emotion_labels[i].lower()] for i, pred in enumerate(predictions)])
    normalized_predictions = scaled_predictions / scaled_predictions.sum()  # Re-normalize to 1
    return emotion_labels[np.argmax(normalized_predictions)], normalized_predictions


# Function to load an image from a URL
def load_image_from_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content))
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError, requests.exceptions.RequestException) as e:
        st.error(f"Error fetching image: {e}")
        return None

# Function to detect and annotate multiple faces in an image
def detect_and_annotate_faces(image):
    faces = detector.detect_faces(image)
    if not faces:
        st.warning("No faces detected.")
        return image, []

    all_confidence_scores = []

    for face in faces:
        x, y, width, height = face["box"]
        roi = image[y:y + height, x:x + width]

        # Predict emotion for the ROI
        emotion, confidence_scores = predict_emotion(roi)
        all_confidence_scores.append(confidence_scores)

        confidence = np.max(confidence_scores)

        confidence_threshold = 0.3
        if confidence < confidence_threshold:
            emotion = "Unknown"

        cv2.rectangle(image, (x, y), (x + width, y + height), (0, 255, 0), 2)
        cv2.putText(image, f"{emotion} ({confidence:.2f})", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    return image, all_confidence_scores

# Function to plot emotion confidence scores using Plotly
def plot_confidence_chart(confidence_scores):
    avg_confidence = np.mean(confidence_scores, axis=0)

    fig = go.Figure(data=[
        go.Bar(x=emotion_labels, y=avg_confidence, marker_color="skyblue")
    ])

    fig.update_layout(
        title="Emotion Prediction Confidence Scores",
        xaxis_title="Emotions",
        yaxis_title="Average Confidence Scores",
        yaxis=dict(range=[0, 1]),
        template="plotly_white"
    )
    return fig


# Page: Image Annotation
if page == "Image Annotation":
    st.header("Upload Image for Annotation")

    # Image upload section
    uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_image is None:
        st.warning("Please upload an image file.")  # Message under the upload button

    # URL input section
    image_url = st.text_input("Or paste image URL here:", "")
    if image_url == "":
        st.warning("Please paste the image address and hit enter.")  # Message under the URL input

    # Check for uploaded image
    if uploaded_image is not None:
        st.success("Image uploaded successfully! Now processing...")  # Success message for upload
        image = Image.open(uploaded_image).convert("RGB")  # Convert to RGB
        image_array = np.array(image)  # Convert to numpy array (now in RGB)

        # Display the original uploaded image
        st.image(image_array, caption='Uploaded Image', use_column_width=True)  # Show original image

        # Detect and annotate faces (original image retains color)
        annotated_image, confidence_scores = detect_and_annotate_faces(image_array)  # Process in RGB
        st.success("Face detection completed! Annotating image...")  # Success message for detection

        # Display annotated image (preserving original RGB color)
        st.image(annotated_image, caption='Annotated Image', use_column_width=True)  # Show annotated image in RGB

        # Plot confidence chart and create download buttons
        confidence_fig = plot_confidence_chart(confidence_scores)
        st.plotly_chart(confidence_fig)  # Use Plotly to show the chart

        # Convert annotated image to bytes for download (keep it in RGB)
        _, annotated_image_buffer = cv2.imencode('.png', cv2.cvtColor(annotated_image,
                                                                      cv2.COLOR_RGB2BGR))  # Convert to BGR for encoding
        annotated_image_bytes = annotated_image_buffer.tobytes()

        # Download button
        st.download_button("Download Annotated Image", annotated_image_bytes, "annotated_image.png", "image/png")
        st.success("Annotated image is ready for download!")  # Success message for download

    # Check for image URL input
    elif image_url:
        st.success("Fetching image from URL...")  # Message indicating URL fetching
        image = load_image_from_url(image_url)
        if image is not None:
            st.success("Image from URL fetched successfully! Now processing...")  # Success for URL fetch

            # Convert to RGB and display the original image
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for display
            st.image(image_rgb, caption='Image from URL', use_column_width=True)  # Show image in RGB

            # Detect and annotate faces (use the original RGB image for detection)
            annotated_image, confidence_scores = detect_and_annotate_faces(image_rgb)  # Keep original RGB
            st.success("Face detection completed! Annotating image...")  # Success message for detection

            # Display the annotated image (preserving original RGB color)
            st.image(annotated_image, caption='Annotated Image', use_column_width=True)  # Show annotated image in RGB

            # Plot confidence chart and create download buttons
            confidence_fig = plot_confidence_chart(confidence_scores)  # Plot confidence scores
            st.plotly_chart(confidence_fig)  # Use Plotly to show the chart

            # Convert annotated image to bytes for download (keep it in RGB)
            _, annotated_image_buffer = cv2.imencode('.png', cv2.cvtColor(annotated_image,
                                                                          cv2.COLOR_RGB2BGR))  # Convert to BGR for encoding
            annotated_image_bytes = annotated_image_buffer.tobytes()

            # Download button
            st.download_button("Download Annotated Image", annotated_image_bytes, "annotated_image.png", "image/png")
            st.success("Annotated image is ready for download!")  # Success message for download
        else:
            st.error("Error: Could not load image from URL.")  # Error message for URL loading

# Video Annotation section

from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

def predict_emotion(roi):
    """Predict the emotion from the given region of interest (ROI)."""
    roi = cv2.resize(roi, (112, 112))  # Resize to match the input shape of your model
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    roi = roi.astype('float32') / 255.0  # Normalize
    roi = np.expand_dims(roi, axis=0)  # Add batch dimension
    roi = np.expand_dims(roi, axis=-1)  # Add channel dimension

    predictions = model.predict(roi)
    emotion_index = np.argmax(predictions[0])
    return emotion_labels[emotion_index]


# Video Annotation section
import cv2
import os
import streamlit as st

# Video Annotation section
if page == "Video Annotation":
    st.header("Upload Video for Annotation")

    uploaded_video = st.file_uploader("Choose a video...", type=["mp4", "mov"])

    # Check if video processing is already in progress
    if "processing" not in st.session_state:
        st.session_state.processing = False

    if uploaded_video is not None and not st.session_state.processing:
        video_path = "temp_video.mp4"
        with open(video_path, "wb") as f:
            f.write(uploaded_video.read())

        st.success("Video uploaded successfully! Now processing...")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error(f"Error: Could not open video {video_path}")
            st.stop()

        # Get the total number of frames for the progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        st.write(f"Total frames to process: {total_frames}")

        # Create a video writer to save the annotated video
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path = "annotated_video.mp4"
        out = cv2.VideoWriter(output_path, fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

        # Create a progress bar
        progress_bar = st.progress(0)
        percentage_text = st.empty()  # Create an empty placeholder for percentage text

        frame_count = 0  # Initialize frame count for progress tracking

        # Mark processing as True
        st.session_state.processing = True

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            faces = detector.detect_faces(frame)
            if faces is None or len(faces) == 0:
                st.warning("No faces detected in this frame.")
                continue

            for face in faces:
                x, y, width, height = face['box']  # Correctly extract the bounding box coordinates

                # Ensure that the bounding box is within frame boundaries
                x, y, width, height = max(0, x), max(0, y), min(frame.shape[1], width), min(frame.shape[0], height)

                # Draw the rectangle around the face
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 1)

                # Extract ROI for emotion prediction
                roi = frame[y:y + height, x:x + width]
                if roi.size > 0:
                    emotion = predict_emotion(roi)

                    # Put text above the rectangle
                    cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            out.write(frame)

            # Update progress bar and percentage text
            frame_count += 1
            progress_percentage = (frame_count / total_frames) * 100
            progress_bar.progress(frame_count / total_frames)
            percentage_text.text(f"Processing: {progress_percentage:.2f}%")  # Display percentage

        cap.release()
        out.release()

        st.success("Video processing completed!")

        with open(output_path, "rb") as f:
            st.download_button("Download Annotated Video", f, output_path, "video/mp4")

        # Cleanup temporary files
        os.remove(video_path)
        os.remove(output_path)

        # Reset processing state after completion
        st.session_state.processing = False

    elif st.session_state.processing:
        st.info("Video processing is currently ongoing. Please wait until it completes.")

import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN

# Load the emotion recognition model
model_path = r"New_model.h5"
model = load_model(model_path)

# Initialize MTCNN for face detection
detector = MTCNN()

# Emotion labels
emotion_labels = ['angry', 'contempt', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Scaling factors for emotions
emotion_scaling = {
    'angry': 1.0,
    'contempt': 1.0,
    'disgust': 1.0,
    'fear': 1.0,
    'happy': 1.0,
    'neutral': 1.0,
    'sad': 1.0,
    'surprise': 1.0,
}

# Function to preprocess the face region for prediction
def preprocess_face(roi):
    roi = cv2.resize(roi, (112, 112))  # Resize according to your model input shape
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)  # Convert to grayscale if needed
    roi = roi.astype('float32') / 255.0  # Normalize pixel values
    roi = np.expand_dims(roi, axis=-1)  # Add channel dimension for grayscale
    roi = np.expand_dims(roi, axis=0)  # Add batch dimension
    return roi

# Function to adjust the predicted probabilities for certain emotions
def adjust_predictions(preds):
    for i, label in enumerate(emotion_labels):
        preds[0][i] *= emotion_scaling[label]  # Apply scaling factor
    preds[0] /= np.sum(preds[0])  # Normalize to ensure probabilities sum to 1
    return preds

# Function to predict emotion from a face ROI
def predict_emotion(roi):
    roi_preprocessed = preprocess_face(roi)
    preds = model.predict(roi_preprocessed)
    preds = adjust_predictions(preds)  # Adjust predictions with the scaling factors
    return preds[0]  # Return the adjusted predictions

# Main function for running the emotion detection
def run_emotion_detection():
    video_capture = cv2.VideoCapture(0)  # Open the default camera
    frame_window = st.empty()  # Placeholder for video frames

    while True:
        ret, frame = video_capture.read()  # Read a frame from the camera
        if not ret:
            break  # If no frame is captured, break the loop

        # Detect faces using MTCNN
        faces = detector.detect_faces(frame)

        for face in faces:
            # Extract bounding box and keypoints (if needed)
            x, y, w, h = face['box']
            roi = frame[y:y + h, x:x + w]  # Extract face region of interest (ROI)

            # Predict the emotion for the ROI
            preds = predict_emotion(roi)
            emotion = emotion_labels[np.argmax(preds)]  # Get the emotion label
            confidence = np.max(preds)  # Get confidence score

            # Apply a confidence threshold to avoid incorrect labeling
            if confidence > 0.6:
                # Draw a rectangle around the face and annotate with the emotion and confidence
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f'{emotion} ({confidence:.2f})', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Display the video frame in the Streamlit app
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB before displaying
        frame_window.image(frame_rgb, channels="RGB", use_column_width=True)

        # Stop the video capture if the button is clicked
        if not st.session_state.start_detection:
            break

    video_capture.release()  # Release the video capture object

    if "start_detection" not in st.session_state:
        st.session_state.start_detection = False  # Initialize the state

    # Toggle button for starting/stopping detection
    if st.button("Start / Stop Detection"):
        st.session_state.start_detection = not st.session_state.start_detection

    if st.session_state.start_detection:
        run_emotion_detection()  # Run the detection if the button is active
