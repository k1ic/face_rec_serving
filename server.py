#coding=utf-8
from gevent import monkey
monkey.patch_all()
import numpy as np
from flask import Flask,request
import os,cv2,json
import tensorflow as tf 
from darkflow.net.build import TFNet
import re,io
import imageio
from multiprocessing.pool import Pool
from PIL import Image
import grequests
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

@app.route('/batch/',methods = ['POST'])
def batch():

    covers = request.form['covers']
    coversArr = covers.split(",")
    tasks=(grequests.get(url) for url in coversArr)
    results=grequests.imap(tasks,size=99)#,size=99
    for re in results:
        print(recFromImg(re))

    return('cool')

   
@app.route('/',methods = ['GET'])
def do_Get():
    url = request.args.get('url', '')
    if re.match(r'^https?:/{2}\w.+$', url):
        print('right url')
        return predictSingle(url)
    else:
        return "Not valid url %s" %url

# async def predict_batch(url):
#     async with grequests.get(url) as resp:
#         response = await resp.content
#         recFromImg(response)
            

def recFromImg(img):
    response = img.content
    byte_stream = io.BytesIO(response)
    roiImg = Image.open(byte_stream)  
    image = np.array(roiImg)
    result = tfnet.return_predict(image,None)
    print(result)
    return result


def predictSingle(url):
    image = imageio.imread(url)
    imgcv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return tfnet.return_predict(imgcv,None)


if __name__ == '__main__':
    app.run(host='10.235.135.12',port=8080)
