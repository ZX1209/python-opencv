
# coding: utf-8

# In[1]:


import os
import cv2
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


os.getcwd(),os.chdir(r'C:\Users\14049\WordAndStudy\Projects\学校\大三上\数字图像处理'),os.getcwd()


# # 读入图像并显示

# In[ ]:


img = cv2.imread(r'./jpg/natasha-taylor-180782-unsplash.jpg')
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# # 修改图像为灰度图并保存图像

# In[4]:


gimg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',gimg)
cv2.waitKey(0)
cv2.destroyAllWindows()


# In[5]:


cv2.imwrite('./jpg/gray-natasha-taylor-180782-unsplash.jpg',gimg)


# # 灰度图像 直方图与装箱直方图

# In[6]:


# ：cv2.calcHist()
# help(cv2.calcHist)
# hist = cv2.calcHist([gimg],[0],None,[256],[0,256])
# plt.plot(hist);plt.show()

plt.hist(gimg.ravel(),256,[0,256]); plt.show()


# In[7]:


hist = cv2.calcHist([gimg],[0],None,[256],[0,256])
sumhist = np.cumsum(hist)
plt.plot(sumhist);plt.show()


# # 读入灰度图像,并显示累计直方图

# In[ ]:


hist = cv2.calcHist([gimg],[0],None,[256],[0,256])
sumhist = np.cumsum(hist)
plt.imshow(gimg)
plt.plot(sumhist);plt.show()


# # 彩色图像 直方图与装箱直方图

# In[24]:


color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    plt.plot(histr,color = col)
    plt.xlim([0,256])
plt.show()


# In[40]:


color = ('b','g','r')
for i,col in enumerate(color):
    histr = cv2.calcHist([img],[i],None,[256],[0,256])
    sumhist = np.cumsum(histr)
    plt.plot(sumhist,color = col)
    plt.xlim([0,256])
plt.show()


# In[26]:


hist = cv2.calcHist([gimg],[0],None,[256],[0,256])


# In[29]:


hist.size,len(hist)


# In[32]:


import numpy as np
sumhist = np.cumsum(hist)


# In[35]:


sumhist.size


# In[36]:


plt.plot(sumhist);plt.show()

