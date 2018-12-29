import numpy as np
global pre_l,pre_r
pre_l = np.loadtxt(r"./pre_left.csv", delimiter=',')
pre_r = np.loadtxt(r"./pre_right.csv", delimiter=',')
pre_l = np.delete(pre_l, [0, 1, 2, 34], axis=0)
pre_r = np.delete(pre_r, [0, 1, 2, 34], axis=0)
def cul2(A):
    area_left = np.zeros(11)
    area_left_sum = np.zeros(11)
    area_right = np.zeros(11)
    area_right_sum = np.zeros(11)
    if len(np.where(A == 1)[0]) != 0:
        Alist = np.where(A == 1)[1]
        Alist_min = float(Alist.min())
        Alist_max = float(Alist.max())
        Aline = np.where(A == 1)[0]
        c_1 = float(Aline.min())
        c_0 = (np.min(Alist[np.where(Aline == c_1)[0]]) + np.max(Alist[np.where(Aline == c_1)[0]])) / 2.0
        a_1 = Aline.max()
        a_0 = (np.min(Alist[np.where(Aline == a_1)[0]]) + np.max(Alist[np.where(Aline == a_1)[0]])) / 2.0
        if np.abs(a_0 - Alist_min) > np.abs(a_0 - Alist_max):
            b_0 = float(Alist.min())
        else:
            b_0 = float(Alist.max())
        b_1 = (np.min(Aline[np.where(Alist == b_0)[0]]) + np.max(Aline[np.where(Alist == b_0)[0]])) / 2.0
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