import cv2
import numpy as np
from clear.DBscan import DB

class Canny:
    def __init__(self,picture):
        self.P = picture
        self.edges = cv2.Canny(picture, 50, 300, L2gradient=False)
        self._find = np.where(self.edges != 0)
    def cov(self):
        for i in np.unique(self._find[0]):
            self.edges[i, np.min(np.where(self.edges[i, ::] != 0)):np.max(np.where(self.edges[i, ::] != 0)) + 1] = 1
        self.image = DB(self.edges)
        self.image[self.image == np.min(self.image)] = 0
        self.image[self.image != 0] = 1
        return self.image
