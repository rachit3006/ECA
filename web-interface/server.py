from copyreg import pickle
import os
from flask import Flask, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['SECRET_KEY'] = 'password'
socket = SocketIO(app)


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@socket.on('message')
def on_message(msg):
    pathcwd = os.getcwd()
    fileName = os.path.join(pathcwd, "data", "latestUserStatus.txt")

    with open(fileName, "r") as f:
        socket.send(f.read())


if __name__ == '__main__':
    app.run(port=3000, debug=True)
