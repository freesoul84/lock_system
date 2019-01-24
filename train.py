#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 10:47:29 2019

@author: alkesha
"""
import cv2,os
import numpy as np


recognizer = cv2.face.LBPHFaceRecognizer_create()

detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def trainer(path):

    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 

    facesamples=[]

    ids=[]

    for imagePath in imagePaths:

        imageread=cv2.imread(imagePath,0)

        imagenp=np.array(imageread,'uint8')
        
        print(os.path.split(imagePath)[-1])

        id_=int(os.path.split(imagePath)[-1].split(".")[-2])

        faces=detector.detectMultiScale(imagenp)

        for (x,y,w,h) in faces:
            facesamples.append(imagenp[y:y+h,x:x+w])
            ids.append(id_)
    return facesamples,ids

face,id_ = trainer('./dataset')
recognizer.train(face, np.array(id_))
recognizer.save('train/train.yml')
