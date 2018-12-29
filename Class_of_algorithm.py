# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 19:10:00 2018

@author: ASUS
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import ndimage
import cv2
from sklearn.cluster import DBSCAN
import time

"""
step1_ways  find the moon
"""


def SAZ(hsv_a):
    hsv_a = cv2.cvtColor(hsv_a, cv2.COLOR_BGR2HSV)
    hsv_a = cv2.inRange(hsv_a, np.array([0, 0, 40]), np.array([200, 255, 255]))
    v = ndimage.median_filter(hsv_a, 8)
    v[v == 255] = 1
    return v


def canny_1(img):
    edges1 = cv2.Canny(img, 50, 300, L2gradient=False)
    find = np.where(edges1 != 0)
    for i in np.unique(find[0]):
        edges1[i, np.min(np.where(edges1[i, ::] != 0)):np.max(np.where(edges1[i, ::] != 0)) + 1] = 1
    return edges1


def canny_2(img):
    edges2 = cv2.Canny(img, 100, 150, L2gradient=True)
    return edges2


def laplace(img):
    laplacian = cv2.Laplacian(img, ddepth=cv2.CV_32F, ksize=17, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    laplacian = cv2.cvtColor(laplacian, cv2.COLOR_BGR2GRAY)
    laplacian[laplacian < 20] = 1
    laplacian[laplacian != 1] = 0

    return laplacian


def sobel(img):
    sobel = cv2.Sobel(img, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=11, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    sobel = cv2.cvtColor(sobel, cv2.COLOR_BGR2GRAY)
    sobel[sobel < 150] = 0
    sobel[sobel != 0] = 1
    return sobel


def scharr(img):
    scharr = cv2.Scharr(img, ddepth=cv2.CV_32F, dx=1, dy=0, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    scharr = cv2.cvtColor(scharr, cv2.COLOR_BGR2GRAY)
    scharr[scharr < 5] = 0
    scharr[scharr != 0] = 1
    return scharr


def erosion(img):
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=2)
    erosion = cv2.cvtColor(erosion, cv2.COLOR_BGR2GRAY)
    erosion[erosion < 30] = 0
    erosion[erosion != 0] = 1
    return erosion


def dilation(img):
    kernel = np.ones((6, 6), np.uint8)
    dilation = cv2.dilate(img, kernel, iterations=2)
    return dilation


def gradient(img):
    kernel = np.ones((6, 6), np.uint8)
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    return gradient


def binary(img):
    th = 200
    max_val = 255
    ret, o = cv2.threshold(img, th, max_val, cv2.THRESH_BINARY)
    return o


"""
step2_ways  separate the moon from the picture
"""


def DB(hsv_a):
    find = np.where(hsv_a == 1)
    line_diff = np.diff(np.sort(np.unique(find[0])))
    list_diff = np.diff(np.sort(np.unique(find[1])))
    if line_diff.max() > 10 or list_diff.max() > 10:
        v_ = np.zeros_like(hsv_a)
        feed = 0
        while line_diff.max() > 10 or list_diff.max() > 10 or feed < 10:
            X = np.array(find).T
            model = DBSCAN(eps=1, min_samples=2)
            model.fit(X)
            no_noise = np.empty(len(np.unique(model.labels_)))
            for i in range(len(np.unique(model.labels_))):
                no_noise[i] = (len(np.where(model.labels_ == i)[0]))
            no = np.where(no_noise == no_noise.max())[0][0]
            a = 1 * (model.labels_ == no)
            b = find[0] * a
            c = find[1] * a
            b = b[np.nonzero(b)[0].min():np.nonzero(b)[0].max() + 1]
            c = c[np.nonzero(c)[0].min():np.nonzero(c)[0].max() + 1]
            for i in range(len(b)):
                v_[b[i], c[i]] = 1
            find = np.where(v_ == 1)
            line_diff = np.diff(np.sort(np.unique(find[0])))
            list_diff = np.diff(np.sort(np.unique(find[1])))
            feed += 1
    else:
        v_ = hsv_a.copy()

    v_ = ndimage.median_filter(v_, 8)
    find = np.where(v_ == 1)

    if len(find[0]) != 0:
        v_ = v_[(np.min(find[0])):(np.max(find[0])), (np.min(find[1])):(np.max(find[1]))]
        length, width = v_.shape
        v_ = cv2.resize(v_, (100, int(100 / float(width) * length)))
    else:
        pass
    return v_


"""
step3_ways  calculate the data
"""


def cul1(A):
    tip = 0
    if len(np.where(A == 1)[0]) != 0:
        Alist = np.where(A == 1)[1]
        Alist_min = Alist.min()
        Alist_max = Alist.max()
        Aline = np.where(A == 1)[0]
        c_1 = Aline.min()
        c_0 = (np.min(Alist[np.where(Aline == c_1)[0]]) + np.max(Alist[np.where(Aline == c_1)[0]])) / 2
        a_1 = Aline.max()
        a_0 = (np.min(Alist[np.where(Aline == a_1)[0]]) + np.max(Alist[np.where(Aline == a_1)[0]])) / 2
        if np.abs(a_0 - Alist_min) > np.abs(a_0 - Alist_max):
            b_0 = Alist.min()
        else:
            b_0 = Alist.max()
        b_1 = (np.min(Aline[np.where(Alist == b_0)[0]]) + np.max(Aline[np.where(Alist == b_0)[0]])) / 2
        circle_j = (-a_1 / 2 + c_1 / 2 + (-a_0 + b_0) * (a_0 / 2 + b_0 / 2) / (a_1 - b_1) - (b_0 - c_0) * (
                    b_0 + c_0) / (2 * (-b_1 + c_1))) / ((-a_0 + b_0) / (a_1 - b_1) - (b_0 - c_0) / (-b_1 + c_1))
        circle_i = a_1 / 2 + b_1 / 2 + (-a_0 + b_0) * (-a_0 / 2 - b_0 / 2 + (
                    -a_1 / 2 + c_1 / 2 + (-a_0 + b_0) * (a_0 / 2 + b_0 / 2) / (a_1 - b_1) - (b_0 - c_0) * (
                        b_0 + c_0) / (2 * (-b_1 + c_1))) / ((-a_0 + b_0) / (a_1 - b_1) - (b_0 - c_0) / (
                    -b_1 + c_1))) / (a_1 - b_1)
        circle_r = np.sqrt((circle_i - a_1) ** 2 + (circle_j - a_0) ** 2)
        return circle_r ** 2 * np.pi, circle_i, circle_j, tip
        if a_0 > Alist_max:
            tip = 1
        else:
            pass
    else:
        circle_r = 100000
        return circle_r ** 2 * np.pi, 50, 50, 1


def cul2(A):
    area_left = np.zeros(11)
    area_left_sum = np.zeros(11)
    area_right = np.zeros(11)
    area_right_sum = np.zeros(11)
    if len(np.where(A == 1)[0]) != 0:
        Alist = np.where(A == 1)[1]
        Alist_min = Alist.min()
        Alist_max = Alist.max()
        Aline = np.where(A == 1)[0]
        c_1 = Aline.min()
        c_0 = (np.min(Alist[np.where(Aline == c_1)[0]]) + np.max(Alist[np.where(Aline == c_1)[0]])) / 2
        a_1 = Aline.max()
        a_0 = (np.min(Alist[np.where(Aline == a_1)[0]]) + np.max(Alist[np.where(Aline == a_1)[0]])) / 2
        if np.abs(a_0 - Alist_min) > np.abs(a_0 - Alist_max):
            b_0 = Alist.min()
        else:
            b_0 = Alist.max()
        b_1 = (np.min(Aline[np.where(Alist == b_0)[0]]) + np.max(Aline[np.where(Alist == b_0)[0]])) / 2
        circle_j = np.floor((-a_1 / 2 + c_1 / 2 + (-a_0 + b_0) * (a_0 / 2 + b_0 / 2) / (a_1 - b_1 + 0.0001) - (
                    b_0 - c_0) * (b_0 + c_0) / (2 * (-b_1 + c_1))) / (
                                        (-a_0 + b_0) / (a_1 - b_1 + 0.0001) - (b_0 - c_0) / (-b_1 + c_1 + 0.0001)))
        circle_i = np.floor(a_1 / 2 + b_1 / 2 + (-a_0 + b_0) * (-a_0 / 2 - b_0 / 2 + (
                    -a_1 / 2 + c_1 / 2 + (-a_0 + b_0) * (a_0 / 2 + b_0 / 2) / (a_1 - b_1 + 0.0001) - (b_0 - c_0) * (
                        b_0 + c_0) / (2 * (-b_1 + c_1))) / ((-a_0 + b_0) / (a_1 - b_1 + 0.0001) - (b_0 - c_0) / (
                    -b_1 + c_1 + 0.0001))) / (a_1 - b_1 + 0.0001))
        circle_r = np.sqrt((circle_i - a_1) ** 2 + (circle_j - a_0) ** 2)
        r = np.floor(circle_r / 10)
        for m in range(11):
            if m == 0:
                area_left_sum[m] = np.floor(np.pi * (r * (m + 1)) ** 2) / 2
                area_right_sum[m] = np.floor(np.pi * (r * (m + 1)) ** 2) / 2
            else:
                area_left_sum[m] = np.floor(np.pi * (r * (m + 1)) ** 2 - np.pi * (r * m) ** 2) / 2
                area_right_sum[m] = np.floor(np.pi * (r * (m + 1)) ** 2 - np.pi * (r * m) ** 2) / 2

        for m in range(len(Alist)):
            i = Aline[m]
            j = Alist[m]
            for k in range(0, 11):
                if np.floor((i - circle_i) ** 2 + (j - circle_j) ** 2) > ((k * r) ** 2) and np.floor(
                        (i - circle_i) ** 2 + (j - circle_j) ** 2) < (((k + 1) * r) ** 2) and A[i, j] == 1:
                    if j < circle_j:
                        area_left[k] += 1
                        break
                    else:
                        area_right[k] += 1
                        break
                else:
                    pass

        precent_left = area_left / area_left_sum
        precent_right = area_right / area_right_sum
    else:
        precent_left = np.zeros(11)
        precent_right = np.zeros(11)
        circle_r = 50
        circle_i = 50
        circle_j = 50

    return precent_left, precent_right, circle_r / 10, circle_i, circle_j


global predict, pre_l, pre_r
predict = np.loadtxt(r"/home/ros-test/Desktop/table.csv", delimiter=',')
predict = np.nan_to_num(predict)
pre_l = np.loadtxt(r"/home/deep/Tian/moon_end/moon/pre_left.csv", delimiter=',')
pre_r = np.loadtxt(r"/home/ros-test/Desktop/right.csv", delimiter=',')
pre_l = np.delete(pre_l, [0, 1, 2, 34], axis=0)
pre_r = np.delete(pre_r, [0, 1, 2, 34], axis=0)

"""
step_3 recognize the moon
"""


def reg_1(v_, data, tip1):
    if v_.sum() < 100:
        pre = 0
    else:
        pre = v_.sum() / float(data)
    if pre > 1:
        pre = 1
    else:
        pass

    choose1 = np.zeros(31)
    for i in range(31):
        choose1[i] = np.abs(pre - predict[i])
    if tip1 == 1:
        choose1[1:11] = 10000
    else:
        choose1[16:32] = 10000
    if len(np.where(choose1 == np.min(choose1))[0]) != 0:
        return "The day is {0}th".format(np.where(choose1 == np.min(choose1))[0][0] + 1), pre
    else:
        return "i don't know", pre


def reg_2(p_l, p_r):
    p_l[p_l > 1] = 1
    p_r[p_r > 1] = 1
    choose2 = np.zeros(31)
    for i in range(31):
        choose2[i] = (((p_l - pre_l[i, ::]) ** 2).sum() + ((p_r - pre_r[i, ::]) ** 2).sum()) / 2.0
    if len(np.where(choose2 == np.min(choose2))[0]) != 0:
        return "The day is {0}th".format(np.where(choose2 == np.min(choose2))[0][0] + 1)
    else:
        return "i don't know"

    """
    step_1 find the moon
    """


preed_l = np.zeros((35, 11))
preed_r = np.zeros((35, 11))
preed = np.zeros(35)
for i in range(5):
    for j in range(7):
        try:
            A = Image.open(r"/home/ros-test/Tian/moon/moon{0}".format(i) + "{0}.jpg".format(j))
            A = np.array(A)
            A_ = A[30:140, 10:len(A[0])]
            start_1 = time.time()

            v_ = erosion(A_)

            if len(v_.shape) > 2:
                v_ = cv2.cvtColor(v_, cv2.COLOR_BGR2GRAY)
                v_[v_ < 100] = 0
                v_[v_ > 0] = 1
            """
            step_2 
            """

            v_ = DB(v_)

            square, circle1_i, circle1_j, tip = cul1(v_)
            t_2, preed[i * 7 + j] = reg_1(v_, square, tip)
            end_1 = time.time()

            start_2 = time.time()
            preed_l[i * 7 + j, ::], preed_r[i * 7 + j, ::], r, circle_i, circle_j = cul2(v_)
            t_1 = reg_2(preed_l[i * 7 + j, ::], preed_r[i * 7 + j, ::])
            end_2 = time.time()

            plt.figure(figsize=[10, 10])
            plt.subplot(221)
            plt.imshow(A)
            plt.subplot(222)
            plt.title('Algorithm_1')

            plt.imshow(v_)

            for k in range(11):
                theta = np.linspace(0, 2 * np.pi, 800)
                x, y = np.cos(theta) * (r * k) + circle_j, np.sin(theta) * (r * k) + circle_i
                plt.plot(x, y, 'b-')

            plt.subplot(223)
            plt.axis("off")

            t = "Algorithm_1"
            plt.text(0, 1, t, ha='left', wrap=True, fontsize=20)
            t = "Avg1_l = {0} ".format("%.4f" % np.mean(preed_l[i * 7 + j, ::]))
            plt.text(-0.1, 0.8, t, ha='left', wrap=True, fontsize=15)
            t = "Avg1_r = {0} ".format("%.4f" % np.mean(preed_r[i * 7 + j, ::]))
            plt.text(0.6, 0.8, t, ha='left', wrap=True, fontsize=15)

            plt.text(0, 0.7, t_1, ha='left', wrap=True, fontsize=15)
            t = "It cost {0}s".format(end_2 - start_2)
            plt.text(0, 0.6, t, ha='left', wrap=True, fontsize=15)
            t = "Algorithm_2"
            plt.text(0, 0.4, t, ha='left', wrap=True, fontsize=20)
            t = "Avg2= {0} ".format("%.4f" % preed[i * 7 + j])
            plt.text(0, 0.2, t, ha='left', wrap=True, fontsize=15)

            plt.text(0, 0.1, t_2, ha='left', wrap=True, fontsize=15)
            t = "It cost {0}s".format(end_1 - start_1)
            plt.text(0, 0.0, t, ha='left', wrap=True, fontsize=15)
            plt.subplot(224)
            plt.title('Algorithm_2')

            plt.imshow(v_)
            theta = np.linspace(0, 2 * np.pi, 800)
            x, y = np.cos(theta) * (r * k) + circle_j, np.sin(theta) * (r * k) + circle_i
            plt.plot(x, y, 'b-')
            plt.savefig(r"/home/ros-test/Tian/moon/table/moon{0}".format(i) + "{0}_table.jpg".format(j))



        except:
            preed_l[i * 7 + j, ::] = np.zeros(11)
            preed_r[i * 7 + j, ::] = np.zeros(11)
            preed[i * 7 + j] = 0
            print("error")





