from numpy import *
import cv2
import numpy as np
from os import listdir, mkdir
from os.path import isdir
from matplotlib import pyplot as plt
import center_algorithms as ca

base_path = './data/'

def load_video(f_name, gr_dims = 0):

    cap = cv2.VideoCapture(f_name)
    ret = True
    frames = []
    while ret:
        ret, img = cap.read() # read one frame from the 'capture' object; img is (H, W, C)
        if ret:
            
            #convert to greyscale
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            width, height = img_gray.shape
            img_gray_resized = img_gray.copy()
            
            if gr_dims == 0:
                gr_dims = width*height

            frames.append(img_gray_resized)
  
    video = np.stack(frames, axis=0) 

    size = video.shape[0]

    if video.size/size != gr_dims:
        return [], gr_dims

    video = video.reshape(size,gr_dims)

    gr_point = np.linalg.qr(video.T)[0][:,:10]

    if size > 10:
        return gr_point, gr_dims
    else:
        return [], gr_dims

data = []
labels = []
start = True
for label in listdir(base_path+'action_youtube_naudio/'):
    if isdir(base_path+'action_youtube_naudio/'+label):
        print('class '+label)
        for sample in listdir(base_path+'action_youtube_naudio/'+label):
            if isdir(base_path+'action_youtube_naudio/'+label+'/'+sample):
                if 'Annotation' not in sample:
                    if start:
                        gr_dims = 0
                        start = False
                    point, gr_dims = load_video(base_path+'action_youtube_naudio/'+label+'/'+sample+'/'+sample+'_01.avi', gr_dims)
                    if len(point)> 1:
                        data.append(point)
                        labels.append(label)
        print(len(data))
        
for c in np.unique(labels):
    mkdir(base_path+'action_youtube_gr/'+c)
i=1
for x, l in zip(data, labels):
    np.save(base_path+'action_youtube_gr/'+l+'/'+l+'_'+str(i)+'.npy', x)
    i+=1