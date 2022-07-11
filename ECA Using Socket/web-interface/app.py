from copyreg import pickle
import json
import os
from pickle import TRUE
from flask import Flask, render_template, Response, request
from PIL import Image
from flask_socketio import SocketIO
from keras.models import load_model
from time import sleep
# from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from tensorflow.keras.utils import img_to_array
from feat import Detector

detector = Detector()

app = Flask(__name__)
app.config['SECRET_KEY'] = 'password'
socket = SocketIO(app)

face_classifier = cv2.CascadeClassifier(r'D:\Rachit\Internship\DBJ\ECA\web-interface\haarcascade_frontalface_default.xml')
classifier = load_model(r'D:\Rachit\Internship\DBJ\ECA\web-interface\model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

def detect_emotion(img_path):

    results = detector.detect_image(img_path)

    emotions_values = {'Anger': results['anger'][0],
                    'Disgust': results['disgust'][0],
                    'Fear': results['fear'][0],
                    'Happiness': results['happiness'][0],
                    'Sadness': results['sadness'][0],
                    'Surprise': results['surprise'][0],
                    'Neutral': results['neutral'][0]}

    emotion = max(emotions_values, key=lambda x: emotions_values[x])

    CI = 0

    if emotion=='Neutral':
        CI = emotions_values['Neutral']*0.9
    elif emotion=='Happiness':
        CI = emotions_values['Happiness']*0.6
    elif emotion=='Surprise':
        CI = emotions_values['Surprise']*0.6
    elif emotion=='Sadness':
        CI = emotions_values['Sadness']*0.3
    elif emotion=='Disgust':
        CI = emotions_values['Disgust']*0.2
    elif emotion=='Anger':
        CI = emotions_values['Anger']*0.25
    else:
        CI = emotions_values['Fear']*0.3

    if CI >= 0.5 and CI <= 1:
        return {'Emotion':emotion, 'Engagement level': 'Engaged'}
    else:
        return {'Emotion':emotion, 'Engagement level': 'Not Engaged'}

def gen_frames(img_path):  
        frame = cv2.imread(img_path)
        
        labels = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)

                prediction = classifier.predict(roi)[0]
                label = emotion_labels[prediction.argmax()]
                label_position = (x, y)
                labels.append(label)
        return labels
        

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/image', methods=['POST'])
def image():
    try:
        image_file = request.files['image']  # get the image
        counter = request.args.get('counter')

        # Set an image confidence threshold value to limit returned data
        threshold = request.form.get('threshold')
        if threshold is None:
            threshold = 0.5
        else:
            threshold = float(threshold)
 
        # finally run the image through tensor flow object detection`
        image_object = Image.open(image_file)
        img_path = os.path.join("images", "download1.jpg")
        image_object = image_object.save(img_path)
        objects = detect_emotion(img_path)
        objects["counter"] = counter
        # labels = gen_frames(img_path)
        # answer = {}
        # answer['Emotion'] = labels[0]
        # answer['Engagement level'] = ""
        return objects
 
    except Exception as e:
        #print('POST /image error: %e' % e)
        return e

# @app.route('/openface', methods=['GET'])
# def openface():
#     return render_template('openface.html')

# @app.route('/mediapipe', methods=['GET'])
# def video_feed():
#     return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/local')
def local():
    return Response(open('./static/local.html').read(), mimetype="text/html")

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST') # Put any other methods you need here
    return response

@socket.on('message')
def on_message(msg):
    pathcwd = os.getcwd()
    fileName = os.path.join(pathcwd, "data", "latestUserStatus.txt")

    with open(fileName, "r") as f:
        socket.send(f.read())


if __name__ == '__main__':
    # app.run(debug=True, host='0.0.0.0', ssl_context=('ssl/server.crt', 'ssl/server.key'))
    app.run(port=3000, debug=True)
