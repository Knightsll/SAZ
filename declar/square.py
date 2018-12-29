import numpy as np
global predict
predict = np.loadtxt(r"./table.csv", delimiter=',')
predict = np.nan_to_num(predict)


def cul1(A):
    tip = 0
    if len(np.where(A == 1)[0]) != 0:
        Alist = np.where(A == 1)[1]
        Alist_min = Alist.min()
        Alist_max = Alist.max()
        Aline = np.where(A == 1)[0]
        c_1 = float(Aline.min())
        c_0 = float((np.min(Alist[np.where(Aline == c_1)[0]]) + np.max(Alist[np.where(Aline == c_1)[0]])) / 2)
        a_1 = float(Aline.max())
        a_0 = float((np.min(Alist[np.where(Aline == a_1)[0]]) + np.max(Alist[np.where(Aline == a_1)[0]])) / 2)
        if np.abs(a_0 - Alist_min) > np.abs(a_0 - Alist_max):
            b_0 = float(Alist.min())
        else:
            b_0 = float(Alist.max())
        b_1 = (np.min(Aline[np.where(Alist == b_0)[0]]) + np.max(Aline[np.where(Alist == b_0)[0]])) / 2
        circle_j = (-a_1 / 2 + c_1 / 2 + (-a_0 + b_0) * (a_0 / 2 + b_0 / 2) / (a_1 - b_1) - (b_0 - c_0) * (b_0 + c_0) / (2 * (-b_1 + c_1))) / ((-a_0 + b_0) / (a_1 - b_1) - (b_0 - c_0) / (-b_1 + c_1))
        circle_i = a_1 / 2 + b_1 / 2 + (-a_0 + b_0) * (-a_0 / 2 - b_0 / 2 + (-a_1 / 2 + c_1 / 2 + (-a_0 + b_0) * (a_0 / 2 + b_0 / 2) / (a_1 - b_1) - (b_0 - c_0) * (b_0 + c_0) / (2 * (-b_1 + c_1))) / ((-a_0 + b_0) / (a_1 - b_1) - (b_0 - c_0) / (-b_1 + c_1))) / (a_1 - b_1)
        circle_r = np.sqrt((circle_i - a_1) ** 2 + (circle_j - a_0) ** 2)
        if a_0 > Alist_max:
            tip = 1
            return circle_r ** 2 * np.pi, circle_i, circle_j, tip
        else:
            return circle_r ** 2 * np.pi, circle_i, circle_j, tip
    else:
        circle_r = 100000
        return circle_r ** 2 * np.pi, 50, 50, 1


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