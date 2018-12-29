import numpy as np
from scipy.misc import imread
import matplotlib.pyplot as plt
from finding.erosion import Erosion
from finding.dilation import Dilation
from finding.canny import Canny
from finding.binary import Binary
from finding.detect_shapes import shape_detection
from finding.contour import Contour
from declar.square import *
from declar.concentric_circle import *
import cv2

np.seterr(divide='ignore',invalid='ignore')

a = cv2.imread(r'test/moon_test_01.jpg')

#use Erosion(image).cov() Dilation(image).cov() the differene is shape_detection(image)
# warning: when we use shape_detection we should use cv2.imread
t = shape_detection(a)

squar, c_i_0, c_j_0, tip = cul1(t)

p_l, p_r, r,c_i_1,c_j_1 = cul2(t)
"""
print(reg_1(t,squar,tip))
print(reg_2(p_l,p_r))
"""
plt.imshow(t)
plt.plot(c_j_0,c_i_0,'bo')
theta = np.linspace(0,2*np.pi,500)
for i in range(1,10):
    plt.plot(r*i*np.cos(theta)+c_j_0,r*i*np.sin(theta)+c_j_0,'o')
plt.show()
