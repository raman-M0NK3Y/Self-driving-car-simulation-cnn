print('Setting UP')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import socketio
import numpy as np
from flask import Flask
import eventlet
#import eventlet.wsgi
import base64
from tensorflow.keras.models import load_model
from io import BytesIO
from PIL import Image 
import cv2

sio = socketio.Server()
app = Flask(__name__)
max_speed = 10

def img_process(image):
    image = image[60:135, :, :]

    #YUV Colourspace - able to better define lane lines and general path
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3),0)
    image = cv2.resize(image, (200,66)) # size used by nvidia
    image = image / 255

    return image

@sio.on('telemetry')
def telemetry(sid, data):
    speed = float(data['speed'])
    image = Image.open(BytesIO(base64.b64decode(data['image']))) # import img coming from car
    image = np.asarray(image)
    image = img_process(image)        #preprocessing just like during training part
    image = np.array([image])
    steering = float(model.predict(image))        #predicting str angle based on img
    throttle = 1.0 - speed / max_speed             #making threshold for speed
    print(f'{steering}, {throttle}, {speed}')
    sendControl(steering, throttle)
    #print('test')



@sio.on('connect')           #connecting and sending commands to simulator
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)         # steering, speed = 0 initially

def sendControl(steering, throttle):
    sio.emit('steer', data={
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()

    })


if __name__ == '__main__':
    model = load_model('model.h5') #load in the model
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app) #communicating w/ port num for this simulator