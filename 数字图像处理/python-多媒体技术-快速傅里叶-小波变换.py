
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
import cv2


# In[2]:


get_ipython().run_line_magic('matplotlib', '')


# In[3]:


os.getcwd(), os.chdir(
    r'C:\Users\14049\WordAndStudy\Projects\学校\大三上\数字图像处理\数字图像处理实验材料\测试图像'), os.getcwd()


# In[4]:


img = cv2.imread(r'lena.bmp', 0)


# In[10]:


f = np.fft.fft2(img)  # 快速傅里叶变换算法得到频率分布
fshift = np.fft.fftshift(f)  # 默认结果中心点位置是在左上角，转移到中间位置

fimg = np.log(np.abs(fshift))  # fft 结果是复数，求绝对值结果才是振幅

# 展示结果
plt.subplot(121), plt.imshow(img, 'gray'), plt.title('Original')
plt.subplot(122), plt.imshow(fimg, 'gray'), plt.title('fft-log')
plt.show()


# In[8]:


ph_f = np.angle(f)
ph_fshift = np.angle(fshift)

plt.subplot(121), plt.imshow(ph_f, 'gray'), plt.title('original')
plt.subplot(122), plt.imshow(ph_fshift, 'gray'), plt.title('center')


# In[9]:


# 逆变换
f1shift = np.fft.ifftshift(fshift)
img_back = np.fft.ifft2(f1shift)
# 出来的是复数，无法显示
img_back = np.abs(img_back)
plt.imshow(img_back, 'gray'), plt.title('img back')
