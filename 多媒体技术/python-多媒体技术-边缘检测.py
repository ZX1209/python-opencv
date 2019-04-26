
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os


# In[2]:


get_ipython().run_line_magic('matplotlib', '')


# In[3]:


os.getcwd(),os.chdir(r'C:\Users\14049\WordAndStudy\Projects\学校\大三上\数字图像处理\数字图像处理实验材料\测试图像'),os.getcwd()


# In[6]:


def cv2ShowImages(imgs):
    for i,img in enumerate(imgs):
        cv2.namedWindow(str(i),cv2.WINDOW_NORMAL)
        cv2.imshow(str(i),img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# In[4]:


image = cv2.imread(r'lena.bmp',0)


# In[7]:


#拉普拉斯边缘检测
lap = cv2.Laplacian(image,cv2.CV_64F)#拉普拉斯边缘检测
lap = np.uint8(np.absolute(lap))##对lap去绝对值

cv2ShowImages([image,lap])


# In[8]:


#Sobel边缘检测
sobelX = cv2.Sobel(image,cv2.CV_64F,1,0)#x方向的梯度
sobelY = cv2.Sobel(image,cv2.CV_64F,0,1)#y方向的梯度

sobelX = np.uint8(np.absolute(sobelX))#x方向梯度的绝对值
sobelY = np.uint8(np.absolute(sobelY))#y方向梯度的绝对值

sobelCombined = cv2.bitwise_or(sobelX,sobelY)#

cv2ShowImages([image,sobelX,sobelY,sobelCombined])


# In[9]:


#Canny边缘检测
canny = cv2.Canny(image,30,150)
cv2ShowImages([image,canny])

