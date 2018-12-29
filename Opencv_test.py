import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imsave
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
    cv2.circle(image, (np.int(((x+x+w)/2)), np.int((y+y+h)/2)), np.int(w/2), (0, 255, 0), 2)
plt.imshow(image)
plt.show()
imsave(r"./output.jpg",image)