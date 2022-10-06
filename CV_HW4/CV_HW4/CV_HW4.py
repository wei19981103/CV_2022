import cv2
import math, sys
import matplotlib.pyplot as plt
import numpy as np


image = cv2.imread('.\\lena.bmp', cv2.IMREAD_GRAYSCALE)
height, width = image.shape
ret, img = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
kernel = [[-2, -1], [-2, 0], [-2, 1],
[-1, -2], [-1, -1], [-1, 0], [-1, 1], [-1, 2],
[0, -2],  [0, -1], [0, 0], [0, 1], [0, 2],
[1, -2],  [1, -1], [1, 0], [1, 1], [1, 2],
          [2, -1], [2, 0], [2, 1]]

kernel_j = [[0, -1], [0, 0], [1, 0]]
kernel_k = [[-1, 0], [-1, 1], [0, 1]]

def dilation(img, se): #se = structuring elements
    result_img = np.zeros([height, width], np.uint8)
    for i in range(height):
        for j in range(width):
            if img[i][j] == 255:
                sum = 0
                for elements in se:
                    if i+elements[0] >= 0 and i+elements[0] <= height-1 and j+elements[1] >= 0 and j+elements[1] <= width-1:
                        sum = sum + img[i+elements[0]][j+elements[1]]
                if sum > 255:
                    for elements in se:
                        result_img[max(0, min(height-1, i+elements[0]))][max(0, min(height-1, j+elements[1]))] = 255
    return result_img

def erosion(img, se):
    result_img = np.zeros([height, width], np.uint8)
    for i in range(height):
        for j in range(width):
            sum = 0
            for elements in se:
                if i+elements[0] >= 0 and i+elements[0] <= height-1 and j+elements[1] >= 0 and j+elements[1] <= width-1:
                    if img[i+elements[0]][j+elements[1]] != 255:
                            sum = 1
            if sum == 0:
                for elements in se:
                    result_img[i][j] = 255
    return result_img

def hit_and_miss(img, se1, se2):
    result_img1 = np.zeros([height, width], np.uint8)
    result_img2 = np.zeros([height, width], np.uint8)
    result_img3 = np.zeros([height, width], np.uint8)
    result_img1 = erosion(img, se1)
    for i in range(height):
        for j in range(width):
                result_img2[i][j] = (img[i][j] - 255) * (-1)
    result_img2 = erosion(result_img2, se2)
    for i in range(height):
        for j in range(width):
            if result_img1[i][j] == 255 and result_img2[i][j] == 255:
                result_img3[i][j] = 255
    return result_img3

###############main####################

dilation_img = dilation(img, kernel)
plt.subplot(1, 5, 1)
plt.imshow(dilation_img, cmap='gray')
plt.title("Dilation")
plt.xticks([]), plt.yticks([])

erosion_img = erosion(img, kernel)
plt.subplot(1, 5, 2)
plt.imshow(erosion_img, cmap='gray')
plt.title("Erosion")
plt.xticks([]), plt.yticks([])

opening_img = dilation(erosion_img, kernel)
plt.subplot(1, 5, 3)
plt.imshow(opening_img, cmap='gray')
plt.title("Opening")
plt.xticks([]), plt.yticks([])

closing_img = erosion(dilation_img, kernel)
plt.subplot(1, 5, 4)
plt.imshow(closing_img, cmap='gray')
plt.title("Closing")
plt.xticks([]), plt.yticks([])

hitnmiss_img = hit_and_miss(img, kernel_j, kernel_k)
plt.subplot(1, 5, 5)
plt.imshow(hitnmiss_img, cmap='gray')
plt.title("Hit & Miss")
plt.xticks([]), plt.yticks([])

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()