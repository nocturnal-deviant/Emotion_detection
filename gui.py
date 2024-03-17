import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np 
import cv2
from tensorflow.keras.models import model_from_json
from PIL import Image, ImageTk

def FacialExmodel(json_file, weights_file):
    with open(json_file, 'r') as file:
        loaded_model_json = file.read()
        model = model_from_json(loaded_model_json)
    
    model.load_weights(weights_file)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def Detect(file_path):
    global label1

    image = cv2.imread(file_path)
    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = facec.detectMultiScale(gray_img, 1.3, 5)

    try:
        for (x,y,w,h) in faces:
            fc = gray_img[y:y+h,x:x+w]
            roi = cv2.resize(fc,(48,48))
            pred = EMOTIONS_LIST[np.argmax(model.predict(roi[np.newaxis,:,:,np.newaxis]))]
            print('Predicted Image is ' + pred)
            label1.configure(foreground="#000000", text="Predicted emotion: " + pred)
    except:
        label1.configure(foreground="#000000", text="Unable to Detect")

def show_detect_button(file_path):
    detect_b = tk.Button(top, text="Detect Emotion", command=lambda: Detect(file_path), padx=10, pady=5, font=('Times', 10, 'bold'))
    detect_b.configure(background="#4CAF50", foreground='white')
    detect_b.place(relx=0.79, rely=0.46)

def upload_image():
    try:
        file_path = filedialog.askopenfilename()
        uploaded = Image.open(file_path)
        uploaded_resized = uploaded.resize((400,400))
        im = ImageTk.PhotoImage(uploaded_resized)

        sign_img.configure(image=im)
        sign_img.image = im
        label1.configure(text='')
        show_detect_button(file_path)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

top = tk.Tk()
top.geometry('800x600')
top.title('Emotion Detector')
top.configure(background='#E0E0E0')

label1 = tk.Label(top, background='#E0E0E0', font=('Times', 15, 'bold'))
sign_img = tk.Label(top)

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = FacialExmodel("model_c.json", "model_.weights.h5")
EMOTIONS_LIST = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

upload = tk.Button(top, text='Upload Image', command=upload_image, padx=10, pady=5, font=('Times', 10, 'bold'))
upload.configure(background="#008CBA", foreground='white')
upload.pack(side='bottom', pady=20)

sign_img.pack(side='bottom', pady=20, padx=20)
label1.pack(side='bottom', pady=10)
heading = tk.Label(top, text='Emotion Detector', pady=20, font=('Times', 25, 'bold'))
heading.configure(background="#E0E0E0", foreground="#000000")
heading.pack()

top.mainloop()
