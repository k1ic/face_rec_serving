#coding=utf-8
import numpy as np
from flask import Flask,request
import os,cv2,json
import tensorflow as tf 
from darkflow.net.build import TFNet
import re
from skimage import io
options = {"model": "./cfg/yolov2-face.cfg", "load": "./bin/weight/yolov2-face.weights", "threshold": 0.4,"labels":"./labels/labels.face","batch":32}
tfnet = TFNet(options)


app = Flask(__name__)

@app.route('/',methods = ['POST'])
def upload():
    f = request.files['file']
    in_memory_file = io.BytesIO()
    f.save(in_memory_file)
    data = np.fromstring(in_memory_file.getvalue(), dtype=np.uint8)
    return tfnet.return_predict(data,None)
   
@app.route('/',methods = ['GET'])
def do_Get():
    url = request.args.get('url', '')
    if re.match(r'^https?:/{2}\w.+$', url):
        print('right url')
        return predict(url)
    else:
        return "Not valid url %s" %url

def predict(url):
    image = io.imread(url)
    imgcv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    cv2img = cv2.imdecode(imgcv,1)
    # findFace(cv2img)

    return tfnet.return_predict(imgcv,None)

if __name__ == '__main__':
    app.run(host='10.235.135.12',port=8080)
