#coding=utf-8
import os
import time
import numpy as np
import tensorflow as tf
import pickle, json
from multiprocessing.pool import ThreadPool

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# config = tf.ConfigProto() 
# config.gpu_options.per_process_gpu_memory_fraction = 0.9 # 占用GPU90%的显存 
# session = tf.Session(config=config)

train_stats = (
    'Training statistics: \n'
    '\tLearning rate : {}\n'
    '\tBatch size    : {}\n'
    '\tEpoch number  : {}\n'
    '\tBackup every  : {}'
)
pool = ThreadPool()

def _save_ckpt(self, step, loss_profile):
    file = '{}-{}{}'
    model = self.meta['name']

    profile = file.format(model, step, '.profile')
    profile = os.path.join(self.FLAGS.backup, profile)
    with open(profile, 'wb') as profile_ckpt: 
        pickle.dump(loss_profile, profile_ckpt)

    ckpt = file.format(model, step, '')
    ckpt = os.path.join(self.FLAGS.backup, ckpt)
    self.say('Checkpoint at step {}'.format(step))
    self.saver.save(self.sess, ckpt)




def return_predict(self, im,img):
    print('return_predict')
    assert isinstance(im, np.ndarray), \
				'Image is not a np.ndarray'
    h, w, _ = im.shape
    im = self.framework.resize_input(im)
    print('h,w',h,w)
    this_inp = np.expand_dims(im, 0)
    print("this_inp")
    feed_dict = {self.inp : this_inp}
    print("feed_dict")
    print(self.out,len(feed_dict))
    print(len(self.sess.run(self.out, feed_dict)))
    out = self.sess.run(self.out, feed_dict)[0]
    print("out")
    _out = out
    boxes = self.framework.findboxes(out)
    if img is not None:
        self.framework.postprocess(self.sess.run(self.out, feed_dict)[0], img,True)
    threshold = self.FLAGS.threshold
    boxesInfo = list()
    print('return_predict',len(boxesInfo))
    for box in boxes:
        tmpBox = self.framework.process_box(box, h, w, threshold)
        if tmpBox is None:
            continue
        """
        {
            "label": tmpBox[4],
            "confidence": tmpBox[6],
            "topleft": {
                "x": tmpBox[0],
                "y": tmpBox[2]},
            "bottomright": {
                "x": tmpBox[1],
                "y": tmpBox[3]}
        }
        """
        boxesInfo.append({
            "label": tmpBox[4],
            "confidence": str(tmpBox[6]),
            "size": {
                "x": str(tmpBox[0]),
                "y": str(tmpBox[2]),
                "w":str(int(tmpBox[1]-tmpBox[0])),
                "h":str(int(tmpBox[3]-tmpBox[2]))}
        })
    print(boxesInfo)
    # print('=====',json.dumps(boxesInfo)) 
    return json.dumps(boxesInfo)

import math

def predict(self):
    inp_path = self.FLAGS.imgdir
    all_inps = os.listdir(inp_path)
    all_inps = [i for i in all_inps if self.framework.is_inp(i)]
    if not all_inps:
        msg = 'Failed to find any images in {} .'
        exit('Error: {}'.format(msg.format(inp_path)))

    batch = min(self.FLAGS.batch, len(all_inps))

    # predict in batches
    n_batch = int(math.ceil(len(all_inps) / batch))
    for j in range(n_batch):
        from_idx = j * batch
        to_idx = min(from_idx + batch, len(all_inps))

        # collect images input in the batch
        inp_feed = list(); new_all = list()
        this_batch = all_inps[from_idx:to_idx]
        for inp in this_batch:
            new_all += [inp]
            this_inp = os.path.join(inp_path, inp)
            this_inp = self.framework.preprocess(this_inp)
            expanded = np.expand_dims(this_inp, 0)
            inp_feed.append(expanded)
        this_batch = new_all

        # Feed to the net
        feed_dict = {self.inp : np.concatenate(inp_feed, 0)}    
        self.say('Forwarding {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        out = self.sess.run(self.out, feed_dict)
        stop = time.time(); last = stop - start
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))

        # Post processing
        self.say('Post processing {} inputs ...'.format(len(inp_feed)))
        start = time.time()
        pool.map(lambda p: (lambda i, prediction:
            self.framework.postprocess(
               prediction, os.path.join(inp_path, this_batch[i])))(*p),
            enumerate(out))
        stop = time.time(); last = stop - start

        # Timing
        self.say('Total time = {}s / {} inps = {} ips'.format(
            last, len(inp_feed), len(inp_feed) / last))
