
# coding: utf-8

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# In[2]:


os.getcwd(),os.chdir(r'C:\Users\14049\WordAndStudy\Projects\学校\大三上\数字图像处理\数字图像处理实验材料\测试图像'),os.getcwd()


# In[3]:


lenaImage = cv2.imread(r'lena.jpg')


# In[4]:


def cv2ShowImages(imgs):
    for i,img in enumerate(imgs):
        cv2.imshow(str(i),img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# # opencv 箱式滤波,中值滤波,高斯滤波

# In[9]:


# blur—图像均值平滑滤波
# blur(src, ksize, dst=None, anchor=None, borderType=None)
blurImages = np.hstack(
    [cv2.blur(lenaImage, (3,3)),
    cv2.blur(lenaImage, (7,7)),
    cv2.blur(lenaImage, (9,9))]
)


# In[10]:


# GaussianBlur—图像高斯平滑滤波
# GaussianBlur(src, ksize, sigmaX, dst=None, sigmaY=None, borderType=None)
GblurImages = np.hstack(
    [cv2.GaussianBlur(lenaImage, (3,3), 0),
     cv2.GaussianBlur(lenaImage, (7,7), 0),
     cv2.GaussianBlur(lenaImage, (9,9), 0)
    ]
)


# In[11]:


# medianBlur—图像中值滤波
# 函数原型：medianBlur(src, ksize, dst=None)
MblurImags = np.hstack(
    [cv2.medianBlur(lenaImage, 3),
     cv2.medianBlur(lenaImage, 7),
     cv2.medianBlur(lenaImage, 9)
    ]
)


# In[12]:


cv2ShowImages([lenaImage,blurImages,GblurImages,MblurImags])


# In[6]:


#中值滤波
blurred = np.hstack([cv2.medianBlur(lenaImage,3),
                     cv2.medianBlur(lenaImage,5),
                     cv2.medianBlur(lenaImage,7)
                     ])

cv2.imshow("Original",lenaImage)

cv2.imshow("Median",blurred)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 自定义高斯滤波函数,实现高斯滤波算法

# In[37]:


guessKernel = np.array([
        [0,1,2,1,0],
        [1,3,5,3,1],
        [2,5,9,5,2],
        [1,3,5,3,1],
        [0,1,2,1,0]],dtype='float32')/57

cImage = cv2.filter2D(lenaImage,-1,guessKernel)

cv2ShowImages([lenaImage,cImage])


# In[41]:


hKernel = np.array([1,1,1,1,1,1,1,1,1],dtype='float32')/9
sKernel = np.array([
    [1],
    [1],
    [1],
    [1],
    [1],
    [1],
    [1]],dtype='float32')/7

hImage = cv2.filter2D(lenaImage,-1,hKernel)
sImage = cv2.filter2D(lenaImage,-1,sKernel)

fImage = sImage = cv2.filter2D(lenaImage,-1,sKernel*hKernel)
cv2ShowImages([lenaImage,hImage,sImage,fImage])


# In[42]:


customKernel = np.array([
        [1,2,3,2,1],
        [2,5,6,5,2],
        [3,6,8,6,3],
         [2,5,6,5,2],
        [1,2,3,2,1]],dtype='float32')
customKernel = customKernel/np.sum(customKernel)

cImage = cv2.filter2D(lenaImage,-1,customKernel)

cv2ShowImages([lenaImage,cImage])

