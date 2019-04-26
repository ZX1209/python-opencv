
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


os.getcwd(),os.chdir(r'C:\Users\14049\WordAndStudy\Projects\学校\大三上\数字图像处理\数字图像处理实验材料\测试图像'),os.getcwd()


# In[5]:


def cv2ShowImages(imgs):
    for i,img in enumerate(imgs):
        cv2.namedWindow(str(i),cv2.WINDOW_NORMAL)
        cv2.imshow(str(i),img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# # 颜色分量旋转

# In[90]:


gimg = cv2.imread(r'lena256rgb.jpg',0)


# In[91]:


rl,cl = gimg.shape


# In[92]:


Y,X = np.meshgrid(np.arange(cl),np.arange(rl))


# In[93]:


Z = gimg


# In[97]:


# This import registers the 3D projection, but is otherwise unused.
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import


plt.rcParams['legend.fontsize'] = 10

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.plot_wireframe(X, Y, Z, label='gray img')

