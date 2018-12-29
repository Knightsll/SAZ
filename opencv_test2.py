import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import signal
from scipy.misc import imsave

global face_cascade
#get face parameters from xml file
face_cascade = cv2.CascadeClassifier(r'/home/deep/anaconda3/envs/sound-recognize/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')


def catch_face(imagepath):
    #read image
    image = imagepath
    gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    #scan the face from the image
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.35,
        minNeighbors=5,
        minSize=(6, 6),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    for(x, y, w, h) in faces:
        cv2.circle(image, (np.int(((x+x+w)/2)), np.int((y+y+h)/2)), np.int(w/2), (255, 255, 255), 3)
    return image

#打开摄像头的方法，window_name为显示窗口名，video_id为你设备摄像头的id，默认为0或-1，如果引用usb可能会改变为1，等
def openvideo(window_name ,video_id):

    cap=cv2.VideoCapture(video_id) # 获取摄像头
    while cap.isOpened():
        ok,frame=cap.read() # ok表示摄像头读取状态，frame表示摄像头读取的图像
        if not ok :
            break
        cv2.imshow('frame',frame) # 将图像矩阵显示在一个窗口中
        if cv2.waitKey(1) &0xFF == ord('q'):
            break

    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("cam closed")
openvideo('mycam' ,0)

    
    
    

