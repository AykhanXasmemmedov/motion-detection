import cv2
import numpy as np

video=cv2.VideoCapture('video/Alibi ALI-IPU3030RV IP Camera Highway Surveillance.mp4')
subtractor1=cv2.createBackgroundSubtractorMOG2(history=100,varThreshold=50)
while video.isOpened():
    ret,frame=video.read()
    roi=frame[250:720,200:900]

    mask=subtractor1.apply(roi)
    _,maskThresh=cv2.threshold(mask,254,255,cv2.THRESH_BINARY)
    contour,_=cv2.findContours(maskThresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        area=cv2.contourArea(cnt)
        if area>6000 and area<10000:
            x,y,w,h=cv2.boundingRect(cnt)
            cv2.rectangle(roi,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.line(roi,(x,y),(x+20,y),(0,255,0),5)
            cv2.line(roi,(x,y),(x,y+20),(0,255,0),5)

            cv2.line(roi,(x+w,y),(x+w-20,y),(0,255,0),5)
            cv2.line(roi,(x+w,y),(x+w,y+20),(0,255,0),5)

            cv2.line(roi,(x,y+h),(x,y+h-20),(0,255,0),5)
            cv2.line(roi,(x,y+h),(x+20,y+h),(0,255,0),5)

            cv2.line(roi,(x+w,y+h),(x+w-20,y+h),(0,255,0),5)
            cv2.line(roi,(x+w,y+h),(x+w,y+h-20),(0,255,0),5)

    cv2.imshow('frame',frame)  
    key=cv2.waitKey(10)
    if key==ord('q'):
        break
video.release()
cv2.destroyAllWindows()