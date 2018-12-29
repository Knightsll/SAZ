import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

from scipy.misc import imsave

'''
#choose image
imagepath = input("Please give me the image path:",)

#get face parameters from xml file
face_cascade = cv2.CascadeClassifier(r'/home/deep/anaconda3/envs/sound-recognize/share/OpenCV/haarcascades/haarcascade_frontalface_alt2.xml')



#read image
image = cv2.imread(imagepath)
gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#scan the face from the image
faces = face_cascade.detectMultiScale(
    gray,
    scaleFactor=1.35,
    minNeighbors=5,
    minSize=(6, 6),
    flags=cv2.CASCADE_SCALE_IMAGE
)

print("\n")
print("find {0} faces!".format(len(faces)))

for(x, y, w, h) in faces:
    cv2.circle(image, (np.int(((x+x+w)/2)), np.int((y+y+h)/2)), np.int(w/2), (255, 255, 255), 3)
'''

#打开摄像头的方法，window_name为显示窗口名，video_id为你设备摄像头的id，默认为0或-1，如果引用usb可能会改变为1，等
def openvideo(window_name ,video_id):

    cap=cv2.VideoCapture(video_id) # 获取摄像头
    while cap.isOpened():
        ok,frame=cap.read() # ok表示摄像头读取状态，frame表示摄像头读取的图像
        if not ok :
            break

        plt.imshow(window_name,frame) # 将图像矩阵显示在一个窗口中
        time.sleep(0.01)
        if  0xFF==ord('q'): # 按键q后break
            break

    # 释放资源
    cap.release()
    print("cam closed")
openvideo('mycam' ,0)

    
    
    

