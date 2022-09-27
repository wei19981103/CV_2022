import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

test = []
cnt = 0

np.set_printoptions(threshold=100000000000)
def union_find(label):
    original_label = label
    cnt = 0
    row, col = cc_img.shape
    global op_cnt
    while label != parent_label[label] and cnt < row * col:
        op_cnt += 1
        label = parent_label[parent_label[label]]
        cnt += 1

    parent_label[original_label] = label # path compression to avoid TLE
    return label

img = cv2.imread(".\\lena.bmp", cv2.IMREAD_GRAYSCALE)
height, width = img.shape
def img_binarize(image):
    #for i in range(height):
    #    for j in range(width):
    #        if image[i][j] >= 128:
    #            image[i][j] = 1
    #        else:
    #            image[i][j] = 0
    return (image > 0x7f) * 0xff

parent_label = []
cc_img = np.zeros([height, width], np.uint32)
img_binarized = img_binarize(img)
cc_img1 = (img_binarized == 0xff) * 1
for i in range (height):
    for j in range(width):
            if img[i][j] >= 128:
                cc_img[i][j] = 1
            else:
                cc_img[i][j] = 0

row, col = cc_img.shape
op_cnt = 0
for i in range(row * col):
    parent_label.append(i)

# do connected components
label = 2
for i in range(row):
    for j in range(col):
        ok1 = 0
        ok2 = 0
        op_cnt += 1
        if cc_img[i, j] == 1:
            if j - 1 >= 0 and cc_img[i, j - 1] > 1: # left has already labeled
                cc_img[i, j] = union_find(cc_img[i, j - 1])
                ok1 = 1

            if i - 1 >= 0 and cc_img[i - 1, j] > 1: # up has already labeled
                if ok1: # set the connected component to make left = up as the same group
                    parent_label[cc_img[i, j]] = union_find(cc_img[i - 1, j])
                else:
                    cc_img[i, j] = cc_img[i - 1, j]

                ok2 = 1

            if ok2 == 0 and ok1 == 0:
                cc_img[i, j] = label
                label += 1

# union and find merging
for i in range(row):
    for j in range(col):
        op_cnt += 1
        if cc_img[i, j] > 1:
            cc_img[i, j] = union_find(cc_img[i, j])

print(cc_img)
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
