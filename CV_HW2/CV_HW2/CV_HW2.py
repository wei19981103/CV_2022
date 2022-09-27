import cv2
import numpy as np
import matplotlib.pyplot as plt

SHIFT = 10
def draw_cent(img,cen_i, cen_j, color):
    cv2.line(img, (cen_j - SHIFT, cen_i), (cen_j + SHIFT, cen_i), color, 2)
    cv2.line(img, (cen_j, cen_i - SHIFT), (cen_j, cen_i + SHIFT), color, 2)


image = cv2.imread(".\\lena.bmp", cv2.IMREAD_GRAYSCALE)
height, width = image.shape

#binarize
bin_img = np.zeros([height, width], np.uint32)
for i in range(height):
    for j in range(width):
        if image[i][j] >= 128:
            bin_img[i][j] = 1
        else:
            bin_img[i][j] = 0

plt.subplot(1, 3, 1)
plt.imshow(bin_img, cmap='gray')
plt.title("binarized")
plt.xticks([]), plt.yticks([])

#draw histogram
hist = [0] * 256
for i in range(height):
        for j in range(width):
            hist[image[i][j]] += 1

plt.subplot(1, 3, 2)
plt.bar(list(range(0,256)), hist, width = 0.5, edgecolor = 'black')
plt.xticks(list(range(0,256,255)))
plt.title("histogram")

#connected components (4-connectivity)

linked = []
for i in range(height * width):
    linked.append([i]) #create a initial list and give initail value to corresponding position ex: linked[20] = [20]

cc_num = 2 # labeling starting from 2 to distinguish labeled or not

#First pass
for i in range(height):
        for j in range(width):
                if bin_img[i][j] != 0:
                    neighbors = [bin_img[i-1][j], bin_img[i][j-1]]

                    if j-1 < 0: #first column
                        neighbors[1] = 0
                    elif i-1 < 0: #first row
                        neighbors[0] = 0

                    if neighbors[0] == 0 and neighbors[1] == 0:
                        bin_img[i][j] = cc_num
                        cc_num += 1

                    else:
                        if neighbors[0] > 1 and neighbors[1] > 1:
                            bin_img[i][j] = min(neighbors)
                        elif neighbors[0] == 0:
                            neighbors[0] = neighbors[1]
                            bin_img[i][j] = neighbors[1]
                        elif neighbors[1] == 0:
                            neighbors[1] = neighbors[0]
                            bin_img[i][j] = neighbors[0]  
                        # first union
                        linked[neighbors[0]] = list(set(linked[neighbors[0]])|set(linked[neighbors[1]]))
                        linked[neighbors[1]] = list(set(linked[neighbors[1]])|set(linked[neighbors[0]]))
#second union
for i in range(1, height-1, 1):
        for j in range(1, width-1, 1):
                if bin_img[i][j] != 0:
                    neighbors = [bin_img[i-1][j], bin_img[i][j-1]]

                    if neighbors[0] > 1 and neighbors[1] > 1:
                        linked[neighbors[0]] = list(set(linked[neighbors[0]])|set(linked[neighbors[1]]))
                        linked[neighbors[1]] = list(set(linked[neighbors[1]])|set(linked[neighbors[0]])) 

#Second pass
cnt = [0]* cc_num
ret, result_img = cv2.threshold(image, 127, 255, 0)
result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
cc_xpos = [[0 for _ in range(1)] for _ in range(cc_num)]
cc_ypos = [[0 for _ in range(1)] for _ in range(cc_num)]

for i in range(height):
    for j in range(width):
        if bin_img[i][j] != 0:
            bin_img[i][j] = min(linked[bin_img[i][j]])
            cnt[bin_img[i][j]] += 1
            #store pixel position
            if cnt[bin_img[i][j]] == 1:
                cc_xpos[bin_img[i][j]][0] = j
                cc_ypos[bin_img[i][j]][0] = i
            else:
                cc_xpos[bin_img[i][j]].append(j)
                cc_ypos[bin_img[i][j]].append(i)

color_rec = (0, 0, 255)
color_cen = (255, 0, 0)
thickness = 3

for i in range(cc_num):
    if cnt[i] >= 500:
        x_sum = 0
        y_sum = 0
        start_point = (min(cc_xpos[i]), min(cc_ypos[i]))
        end_point = (max(cc_xpos[i]), max(cc_ypos[i]))
        cv2.rectangle(result_img, start_point, end_point, color_rec, thickness)
        for xcord in cc_xpos[i]:
            x_sum = xcord + x_sum
        for ycord in cc_ypos[i]:
            y_sum = ycord + y_sum
        x_cen = round(x_sum / (len(cc_xpos[i])-1))
        y_cen = round(y_sum / (len(cc_ypos[i])-1))
        draw_cent(result_img, y_cen, x_cen, color_cen)


plt.subplot(1, 3, 3)
plt.imshow(result_img)
plt.title("Connected Components")
plt.xticks([]), plt.yticks([])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
