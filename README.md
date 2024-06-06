# Emotion_detection
Simple Emotion_Detection tool using machine learning

I've created a Python-based graphical user interface (GUI) application that can detect facial emotions from images. This application uses Tkinter for the GUI, OpenCV for image processing, and a pre-trained deep learning model for recognizing emotions.

Key Components and Workflow:
Model Loading:

I have a function FacialExmodel(json_file, weights_file) that loads a pre-trained facial emotion recognition model from a JSON file and its corresponding weights file. I compile the model using the Adam optimizer and categorical crossentropy loss function.
Image Detection:

The Detect(file_path) function reads the uploaded image, converts it to grayscale, detects faces within the image using a Haar Cascade Classifier (haarcascade_frontalface_default.xml), and predicts the emotion of the detected face.
The detected emotion is then displayed on the GUI.
GUI Elements:

Using Tkinter, I built the application with a button to upload an image (Upload Image), a label to display the uploaded image, and another label to show the predicted emotion.
When an image is uploaded, it is displayed on the GUI, and a "Detect Emotion" button appears. Clicking this button triggers the emotion detection process.
Emotion Detection:

In the Detect function, the image is processed to detect faces. It extracts the region of interest (ROI) corresponding to the face, resizes it to the input size expected by the model (48x48), and predicts the emotion using the loaded model.
The predicted emotion is displayed on the GUI.
Error Handling:

I included error handling to manage exceptions during image upload and emotion detection, displaying error messages if something goes wrong.
Outcome:
The outcome of this application is a user-friendly interface that allows users to upload an image, detect faces in the image, and predict the emotion of the detected face using a deep learning model. The detected emotion is displayed on the screen, providing immediate feedback. This application can be used for mood analysis, emotion recognition research, or interactive applications that respond to users' emotional states.  
