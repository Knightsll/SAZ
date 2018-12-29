import numpy as np
from clear.DBscan import DB

np.seterr(divide='ignore',invalid='ignore')
class Contour:
    def __init__(self, picture):
        #Change RGB to HSV
        self.P = picture
        self.V = np.max(self.P, axis=2)
        self._min_P = np.min(self.P, axis=2)
        self.choose = (self.V != 0)+0
        self.V[self.V == 0] = 1
        self.S = (self.V - np.min(self.P,axis=2))/self.V
        self.S = self.S*self.choose
        self.H = ((self.V == self.P[::, ::, 0])+0)*(60*(self.P[::, ::, 1]-self.P[::, ::, 2])/(self.V - self._min_P))
        self.H = self.H + ((self.V == self.P[::, ::, 1])+0)*(120 + 60*(self.P[::, ::, 2] - self.P[::, ::, 0])/(self.V - self._min_P))
        self.H = self.H + ((self.V == self.P[::, ::, 2])+0)*(240 + 60*(self.P[::, ::, 0] - self.P[::, ::, 1])/(self.V - self._min_P))
        self.image = ((self.H < 200) + 0) * ((self.S < 80) + 0) * ((self.V > 140) + 0)
    def cov(self):

        self.img = DB(self.image)
        self.img[self.img==np.min(self.img)]=0
        self.img[self.img!=0]=1
        return self.img

