
# coding: utf-8

# In[3]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


# In[4]:


get_ipython().run_line_magic('matplotlib', '')


# In[5]:


get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[6]:


os.getcwd(),os.chdir(r'C:\Users\14049\WordAndStudy\Projects\学校\大三上\数字图像处理\数字图像处理实验材料\测试图像'),os.getcwd()


# In[7]:


def cv2ShowImages(imgs):
    for i,img in enumerate(imgs):
        cv2.namedWindow(str(i),cv2.WINDOW_NORMAL)
        cv2.imshow(str(i),img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# # 颜色分量旋转

# In[8]:


img = cv2.imread(r'baboon.bmp')


# In[9]:


b,g,r = cv2.split(img)


# In[10]:


#                              b g r
#                               | | |
cimg = cv2.merge((g,r,b))


# In[11]:


cv2ShowImages([img,cimg])

