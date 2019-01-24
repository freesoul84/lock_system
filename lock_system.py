#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 10:55:28 2019

@author: alkesha
"""

import cv2
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('train/train.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)

cam = cv2.VideoCapture(0)

while True:
    ret, frame =cam.read()
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces=faceCascade.detectMultiScale(gray, 1.2,3)
    
    for(x,y,w,h) in faces:
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(225,0,0),2)
        
        Id, c = recognizer.predict(gray[y:y+h,x:x+w])
        
        print(c)
        
        if(c>95):
            
            if(Id==1):
            
                Id="alia"
            
            elif(Id==2):
            
                Id="disha"
            
            cv2.putText(frame, str("Unlock"), (x,y+200),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 2)
        
        else:
            
            Id="Unknown"
            
            cv2.putText(frame, str("lock"), (x+50,y+300),cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        cv2.putText(frame, str(Id), (x,y-40), font, 2, (255,255,255), 2)
    
    cv2.imshow('system',frame) 
    
    if cv2.waitKey(1)==ord('q'):
    
        break

cam.release()

cv2.destroyAllWindows()
