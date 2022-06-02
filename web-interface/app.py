from copyreg import pickle
import os
from flask import Flask, render_template, Response
from flask_socketio import SocketIO
from keras.models import load_model
from time import sleep
# from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
from tensorflow.keras.utils import img_to_array

app = Flask(__name__)
app.config['SECRET_KEY'] = 'password'
socket = SocketIO(app)

face_classifier = cv2.CascadeClassifier(r'D:\Rachit\Internship\DBJ\ECA\web-interface\haarcascade_frontalface_default.xml')
classifier = load_model(r'D:\Rachit\Internship\DBJ\ECA\web-interface\model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

cap = cv2.VideoCapture(0)

def gen_frames():  
    while True:
        success, frame = cap.read()  # read the camera frame
        if not success:
            break
        else:
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
                    cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, 'No Faces', (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            #cv2.imshow('Emotion Detector', frame)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


@app.route('/openface', methods=['GET'])
def index():
    return render_template('openface.html')

@app.route('/mediapipe', methods=['GET'])
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socket.on('message')
def on_message(msg):
    pathcwd = os.getcwd()
    fileName = os.path.join(pathcwd, "data", "latestUserStatus.txt")

    with open(fileName, "r") as f:
        socket.send(f.read())


if __name__ == '__main__':
    app.run(port=3000, debug=True)
