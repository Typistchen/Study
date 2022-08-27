import cv2
import matplotlib.pyplot as plt
import numpy as np
img = cv2.imread('D:\\WorkRoom\\postgraduate\\Study\\Day2\\1.png')
top_size, bottom_size, left_size, right_size = (50, 50, 50, 50)
cv2.imshow('image', img)
cv2.waitKey(0)
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, right_size, left_size, borderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, right_size, left_size, borderType=cv2.BORDER_REFLECT)
reflect01 = cv2.copyMakeBorder(img, top_size, bottom_size, right_size, left_size, borderType=cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, right_size, left_size, borderType=cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img, top_size, bottom_size, right_size, left_size, borderType=cv2.BORDER_CONSTANT, value=0)

plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect01, 'gray'), plt.title('REFELECT01')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
plt.show()


# 阈值
img2 = cv2.imread('D:\\WorkRoom\\postgraduate\\Study\\Day2\\1.png', cv2.IMREAD_GRAYSCALE )
ret, thresh1 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img2, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img2, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img2, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Oroginal Image', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img2, thresh1, thresh2, thresh3, thresh4, thresh5]
for i in range(6):
    plt.subplot(2, 3, i+1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
# 均值滤波
blur = cv2.blur(img, (3,3))

cv2.imshow('blur', blur)
cv2.waitKey(0)

# 高斯滤波
aussion = cv2.GaussianBlur(img, (5, 5), 1)
cv2.imshow('sussion', aussion)
cv2.waitKey(0)

# 腐蚀操作



