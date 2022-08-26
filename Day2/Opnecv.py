import cv2
import matplotlib.pyplot as plt
import numpy as np

# 图片的读取
img = cv2.imread('1.png')
img2 = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE )
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.imshow('image', img2)
cv2.waitKey(0)
# cv2.destoryAllWindows()
print(img)
print('img.shape:', img.shape)
print(img2)
print('img.shape:', img2.shape)