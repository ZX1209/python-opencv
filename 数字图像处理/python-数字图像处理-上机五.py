import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

get_ipython().run_line_magic('matplotlib', '')

get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")

os.getcwd(), os.chdir(
    r'C:\Users\14049\WordAndStudy\Projects\学校\大三上\数字图像处理\数字图像处理实验材料\测试图像'), os.getcwd()


def cv2ImageShow(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cv2ShowImages(imgs):
    for i, img in enumerate(imgs):
        cv2.namedWindow(str(i), cv2.WINDOW_NORMAL)
        cv2.imshow(str(i), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread(r'carnev256gray.png', 0)


hist = cv2.calcHist([img], [0], None, [256], [0, 256])


Hist = np.cumsum(hist)


XHist = np.cumsum(hist**0.5)


plt.gray()


imgList1 = img.tolist()
w, h = img.shape
M = w*h
K = 256
for v in range(h):
    for u in range(w):
        a = imgList1[v][u]
        b = (Hist[a]*(K-1))//M
        imgList1[v][u] = b


tmp = np.histogram(imgList1, bins=np.arange(256))
tmpcum1 = np.cumsum(tmp[0])


imgList2 = img.tolist()
w, h = img.shape
M = w*h
K = 256
for v in range(h):
    for u in range(w):
        a = imgList2[v][u]
        b = (XHist[a]*(K-1))//XHist[255]
        imgList2[v][u] = b


tmp = np.histogram(imgList2, bins=np.arange(256))
tmpcum2 = np.cumsum(tmp[0])


cv2ShowImages([np.array(imgList1, dtype='uint8'),
               np.array(imgList2, dtype='uint8')])

plt.subplot(121)
plt.plot(tmpcum1)
plt.title('equalization hist')
plt.subplot(122)
plt.plot(tmpcum2)
plt.title('modified equalization hist')

plt.show()
