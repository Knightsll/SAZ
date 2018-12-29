import cv2
import numpy as np

class Binary:
    def __init__(self,picture):
        self.P = picture
        self._th = 160
        self._max_val = 255
    def cov(self):
        self.ret, self.o = cv2.threshold(self.P, self._th, self._max_val, cv2.THRESH_BINARY)
        self.o = cv2.cvtColor(self.o, cv2.COLOR_BGR2GRAY)
        self.o[self.o<60]=0
        self.o[self.o!=0]=1
        return self.o