import cv2
import numpy as np

def mean_percent_bright(image, kernel_size = 5):
    blur_img = cv2.blur(image, (kernel_size, kernel_size))
    mean_bright = np.mean(blur_img)
    mean_percent = mean_bright / 255.0
    return mean_percent

def variance_of_laplacian(image):
	return cv2.Laplacian(image, cv2.CV_64F).var()