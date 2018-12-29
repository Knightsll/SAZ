from skimage import transform
import cv2
import numpy as np
from scipy.misc import imread
from scipy import ndimage


def shape_detection(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	resized = transform.resize(image, (100,int(100.0*np.shape(image)[1]/np.shape(image)[0])))
	ratio = image.shape[0] / float(resized.shape[0])

	# convert the resized image to grayscale, blur it slightly,
	# and threshold it

	blurred = cv2.GaussianBlur(gray, (5, 5), 0)
	thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)[1]

	# find contours in the thresholded image and initialize the
	# shape detector
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

	def cov(image, t):
		cnt = cnts[t]

		# loop over the contours
		for c in cnt:
			# compute the center of the contour, then detect the name of the
			# shape using only the contour
			M = cv2.moments(c)
			cX = int((M["m10"] / M["m00"]))
			cY = int((M["m01"] / M["m00"]))
			c = c.astype("float")
			c = c.astype("int")
		i_max = np.max(c[::, ::, 0])
		i_min = np.min(c[::, ::, 0])
		j_max = np.max(c[::, ::, 1])
		j_min = np.min(c[::, ::, 1])
		img = gray[j_min:(j_max+1), i_min:(i_max+1)]
		img[img < 100] = 0
		img[img != 0] = 1
		img = transform.resize(img, (100, int(100.0 * np.shape(img)[1] / np.shape(img)[0])))
		img[img == np.max(img)] = 1
		img[img == np.min(img)] = 0
		img = ndimage.median_filter(img, 8)
		return img
	try:
		return cov(gray, 1)
	except:
		try :
			return cov(gray, 0)
		except:
			return np.zeros((100, 100))

