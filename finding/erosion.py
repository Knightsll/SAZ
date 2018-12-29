import cv2
import numpy as np
from clear.DBscan import DB

class Erosion:
    def __init__(self, picture):
        self.P = picture
        self.kernel = np.ones((3, 3), np.uint8)
        self.erosion = cv2.erode(self.P, self.kernel, iterations=2)
    def cov(self):
        self.erosion = cv2.cvtColor(self.erosion, cv2.COLOR_BGR2GRAY)
        self.erosion[self.erosion < 60] = 0
        self.erosion[self.erosion != 0] = 1
        self.image = DB(self.erosion)
        self.image[self.image == np.min(self.image)] = 0
        self.image[self.image != 0] = 1
        return self.image

