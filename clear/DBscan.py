import numpy as np
from sklearn.cluster import DBSCAN
from scipy import ndimage
from skimage import transform

def DB(hsv_a):
    hsv_a = ndimage.median_filter(hsv_a, 8)
    find = np.where(hsv_a == 1)
    line_diff = np.diff(np.sort(np.unique(find[0])))
    list_diff = np.diff(np.sort(np.unique(find[1])))
    try:
        if line_diff.max() > 10 or list_diff.max() > 10:
            v_ = np.zeros_like(hsv_a)
            feed = 0
            t = 0
            while line_diff.max() > 10 or list_diff.max() > 10 or feed < 10:
                if t<10:
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
                    t+=1
                else:
                    break
        else:
            v_ = hsv_a.copy()
    except:
        v_ = hsv_a.copy()


    find = np.where(v_ == 1)

    if len(find[0]) != 0:
        v_ = v_[(np.min(find[0])):(np.max(find[0])), (np.min(find[1])):(np.max(find[1]))]
        length, width = v_.shape
        v_ = transform.resize(v_, (100, int(100 / float(width) * length)))
    else:
        pass
    return v_