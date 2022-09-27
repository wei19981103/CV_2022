import cv2
import numpy as np
import matplotlib.pyplot as plt

#create histogram
def draw_hist(img):
    hist = [0] * 256
    height, width = img.shape
    for i in range(height):
        for j in range(width):
            hist[img[i][j]] += 1
    return hist

#original img and its histogram
image = cv2.imread(".\\lena.bmp", cv2.IMREAD_GRAYSCALE)
height, width = image.shape
hist_original = draw_hist(image)

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original")
plt.xticks([]), plt.yticks([])

plt.subplot(2, 3, 4)
plt.bar(list(range(0,256)), hist_original, width = 0.5, edgecolor = 'black')
plt.xticks(list(range(0,256,50)))

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()