
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


get_ipython().run_line_magic('matplotlib', '')


# In[3]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[4]:


os.getcwd(), os.chdir(
    r'C:\Users\14049\WordAndStudy\Projects\学校\大三上\数字图像处理\数字图像处理实验材料\测试图像'), os.getcwd()


# In[5]:


def cv2ShowImages(imgs):
    for i, img in enumerate(imgs):
        cv2.namedWindow(str(i), cv2.WINDOW_NORMAL)
        cv2.imshow(str(i), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# # 灰度图像最大值和最小值滤波算法

# In[27]:


img = cv2.imread(r'lena(slat pepper 0.05).bmp', 0)


# In[28]:


imgList = img.tolist()


# In[29]:


rl = len(imgList)
cl = len(imgList[0])
dis = 3
for ri in range(rl):
    for ci in range(cl):
        topBound = ri-dis if ri-dis > 0 else 0
        bottomBound = ri+dis if ri+dis < rl else rl-dis
        leftBound = ci-dis if ci-dis > 0 else 0
        rightBound = ci+dis if ci+dis < cl else cl-dis

        imgList[ri][ci] = int(
            np.min(img[topBound:bottomBound+1, leftBound:rightBound+1]))

# 数值类型!!!
minImageArray = np.array(imgList, 'uint8')


# In[30]:


rl = len(imgList)
cl = len(imgList[0])
dis = 3
for ri in range(rl):
    for ci in range(cl):
        topBound = ri-dis if ri-dis > 0 else 0
        bottomBound = ri+dis if ri+dis < rl else rl-dis
        leftBound = ci-dis if ci-dis > 0 else 0
        rightBound = ci+dis if ci+dis < cl else cl-dis

        imgList[ri][ci] = int(
            np.max(img[topBound:bottomBound+1, leftBound:rightBound+1]))

# 数值类型!!!
maxImgArray = np.array(imgList, 'uint8')


# In[50]:


cv2ShowImages([img, minImageArray, maxImgArray])


# # 彩色图像 最大,最小值滤波

# In[53]:


colorimg = cv2.imread(r'lena.jpg')


# In[65]:


kernel = np.ones((5, 5), 'uint8')
dst1 = cv2.dilate(colorimg, kernel)


# In[66]:


kernel = np.ones((5, 5), 'uint8')
dst2 = cv2.erode(colorimg, kernel)


# In[67]:


cv2ShowImages([colorimg, dst1, dst2])


# # 滑块

# In[ ]:


# tmpimg = cv2.imread(r'lena.jpg')

# cv2.namedWindow('image')

# # 创建滑块,注册回调函数
# cv2.createTrackbar('R','image',0,255,do_nothing)
# cv2.createTrackbar('G','image',0,255,do_nothing)
# cv2.createTrackbar('B','image',0,255,do_nothing)


# r = cv2.getTrackbarPos('R','image')
# g = cv2.getTrackbarPos('G','image')
# b = cv2.getTrackbarPos('B','image')

# while True:
#     k = cv2.waitKey(1) & 0xFF
#     if k == 8:
#         break
#     r = cv2.getTrackbarPos('R','image')
#     g = cv2.getTrackbarPos('G','image')
#     b = cv2.getTrackbarPos('B','image')


# cv2.waitKey(0)
# cv2.destroyAllWindows()
