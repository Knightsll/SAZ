import cv2
import numpy as np
from clear.DBscan import DB

class Dilation:
    def __init__(self, picture):
        self.P = picture
        self.kernel = np.ones((6, 6), np.uint8)
    def cov(self):
        self.dilatiion = cv2.dilate(self.P, self.kernel, iterations=2)
        self.dilatiion = cv2.cvtColor(self.dilatiion, cv2.COLOR_BGR2GRAY)
        self.dilatiion[self.dilatiion<60]=0
        self.dilatiion[self.dilatiion!=0]=1
        self.image = DB(self.dilatiion)
        self.image[self.image == np.min(self.image)] = 0
        self.image[self.image != 0] = 1
        return self.image