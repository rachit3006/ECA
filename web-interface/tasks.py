from keras.models import load_model
import cv2
import numpy as np
from tensorflow.keras.utils import img_to_array
import os

face_classifier = cv2.CascadeClassifier(r'D:\Rachit\Internship\DBJ\ECA\web-interface\haarcascade_frontalface_default.xml')
classifier = load_model(r'D:\Rachit\Internship\DBJ\ECA\web-interface\model.h5')

emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

def get_results(img_path):  
        return True
        # frame = cv2.imread(os.path.join(img_path, "download.jpg"))
        
        # labels = []
        # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # faces = face_classifier.detectMultiScale(gray)

        # for (x, y, w, h) in faces:
        #     cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        #     roi_gray = gray[y:y + h, x:x + w]
        #     roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        #     if np.sum([roi_gray]) != 0:
        #         roi = roi_gray.astype('float') / 255.0
        #         roi = img_to_array(roi)
        #         roi = np.expand_dims(roi, axis=0)

        #         prediction = classifier.predict(roi)[0]
        #         label = emotion_labels[prediction.argmax()]
        #         label_position = (x, y)
        #         labels.append(label)

        #         emotions_values = {'Anger': prediction[0],
        #                     'Disgust': prediction[1],
        #                     'Fear': prediction[2],
        #                     'Happiness': prediction[3],
        #                     'Neutral': prediction[4],
        #                     'Sadness': prediction[5],
        #                     'Surprise': prediction[6]
        #                     }

        #         emotion = max(emotions_values, key=lambda x: emotions_values[x])

        #         CI = 0

        #         if emotion=='Neutral':
        #             CI = emotions_values['Neutral']*0.9
        #         elif emotion=='Happiness':
        #             CI = emotions_values['Happiness']*0.6
        #         elif emotion=='Surprise':
        #             CI = emotions_values['Surprise']*0.6
        #         elif emotion=='Sadness':
        #             CI = emotions_values['Sadness']*0.3
        #         elif emotion=='Disgust':
        #             CI = emotions_values['Disgust']*0.2
        #         elif emotion=='Anger':
        #             CI = emotions_values['Anger']*0.25
        #         else:
        #             CI = emotions_values['Fear']*0.3

        #         if CI >= 0.5 and CI <= 1:
        #             engagement = 'Engaged'
        #         else:
        #             engagement = 'Not Engaged'
        
        #         os.remove(os.path.join(img_path, "download.jpg"))
        #         os.rmdir(img_path)
        #         return {"emotion": emotion, "engagement": engagement}