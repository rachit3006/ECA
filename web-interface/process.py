from flask import Flask, render_template, Response, request
from PIL import Image
from flask_socketio import SocketIO, emit
import time
import io
import base64
import numpy as np
from engineio.payload import Payload
import os
import secrets
from tasks import get_results
import redis
from rq import Queue

r = redis.Redis()
q = Queue(connection = r)

Payload.max_decode_packets = 2048

app = Flask(__name__)
socketio = SocketIO(app,cors_allowed_origins='*' )

app.config["UPLOAD_DIR"] = "D:\Rachit\Internship\DBJ\ECA\web-interface\images"

@app.route('/', methods=['POST', 'GET'])
def index():
    return render_template('index1.html')

def readb64(base64_string):
    image_dir_name = secrets.token_hex(16)
    os.mkdir(os.path.join(app.config["UPLOAD_DIR"], image_dir_name))
    
    idx = base64_string.find('base64,')
    base64_string  = base64_string[idx+7:]

    sbuf = io.BytesIO()

    sbuf.write(base64.b64decode(base64_string, ' /'))
    
    pimg = Image.open(sbuf)
    pimg.save(os.path.join(app.config["UPLOAD_DIR"], image_dir_name, "download.jpg"))

    return os.path.join(app.config["UPLOAD_DIR"], image_dir_name)

def moving_average(x):
    return np.mean(x)

global fps,prev_recv_time,cnt,fps_array
fps=30
prev_recv_time = 0
cnt=0
fps_array=[0]

@socketio.on('image')
def image(data_image):
    global fps,cnt, prev_recv_time,fps_array
    recv_time = time.time()
    text  =  'FPS: '+str(fps)
    img_path = readb64(data_image)

    job = q.enqueue_call(func=get_results, args=(img_path,), result_ttl=5000)

    # emit the frame back
    emit('response_back', job)
    
    fps = 1/(recv_time - prev_recv_time)
    fps_array.append(fps)
    fps = round(moving_average(np.array(fps_array)),1)
    prev_recv_time = recv_time
    #print(fps_array)
    cnt+=1
    if cnt==30:
        fps_array=[fps]
        cnt=0
    
if __name__ == '__main__':
    socketio.run(app,port=3000 ,debug=True)
   

