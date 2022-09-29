import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

#vairables

image = cv2.imread(".\\lena.bmp", 1)
size = image.shape
height = size[0]
width = size[1]
center_x, center_y = (width / 2, height / 2)

#upsid-down
img1 = np.zeros([height, width, 3], np.uint8)
for i in range(height):
    for j in range(width):
        img1[i][j] = image[height - 1 - i][j]

#right-side-left
img2 = np.zeros([height, width, 3], np.uint8)
for i in range(height):
    for j in range(width):
        img2[i][j] = image[i][width - 1 - j]

#diagonally flip
img3 = np.zeros([height, width, 3], np.uint8)
for i in range(height):
    for j in range(width):
        img3[i][j] = image[height - 1 - i][width - 1 - j]

#rotate 45 degrees clockwise
img4 = np.zeros([height, width, 3], np.uint8)
rads = math.pi / 4
for i in range(height):
    for j in range(width):
        rotate_i = -(j - center_x) * math.sin(rads) + (i - center_y) * math.cos(rads)
        rotate_j = (i - center_y) * math.sin(rads) + (j - center_x) * math.cos(rads)

        rotate_i = int(round(rotate_i) + center_y)
        rotate_j = int(round(rotate_j) + center_x)

        if (rotate_i >= 0 and rotate_j >= 0 and rotate_i < height and rotate_j < width):
            img4[i][j] = image[rotate_i][rotate_j]

#shirnk in half
img5 = np.zeros([height, width, 3], np.uint8)
scale = 0.5
for i in range(height):
    for j in range(width):
        scale_i = center_y + (i - center_y) * scale
        scale_j = center_x + (j - center_x) * scale
        scale_i = round(scale_i)
        scale_j = round(scale_j)
        img5[scale_i][scale_j] = image[i][j]

#binarize at 128
img6 = np.zeros([height, width, 3], np.uint8)
gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
for i in range(height):
    for j in range(width):
        if gray_img[i][j] >= 128:
            img6[i][j] = (255, 255, 255)
        else:
            img6[i][j] = (0, 0, 0)

titles = ['upside-down', 'right-side-left', 'diagonally flip', 'rotate 45 cw', 'shrink in half', 'binarize at 128']
result_img = [img1, img2, img3, img4, img5, img6]


#create window
for x in range(len(titles)):
    plt.subplot(2, 3, x + 1)
    plt.imshow(result_img[x])
    plt.title(titles[x])
    plt.xticks([]), plt.yticks([])   
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
