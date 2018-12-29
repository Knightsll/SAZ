#!/usr/bin/env python
import cv2
from distutils.version import LooseVersion

if LooseVersion(cv2.__version__).version[0] == 2:
    pass
else:
    #choose image
    imagepath = raw_input("Please give me the image path:",)
           
    #get face parameters from xml file
    xmlpath = '/opt/ros/kinetic/share/OpenCV-3.3.1-dev/haarcascades/haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(xmlpath)
    #read image
    image = cv2.imread(imagepath)
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    #scan the face from the image
    faces=face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.15,
        minNeighbors=10,
        minSize=(50, 50)
    )

    print "\n"
    print "find {0} faces!".format(len(faces))

    for(x,y,w,h) in faces:
        cv2.circle(image,((x+x+w)/2,(y+y+h)/2),w/2,(0,255,0),2)
    
    cv2.imwrite("/home/ros-test/test_opencv.jpg",image)
    
    
    

