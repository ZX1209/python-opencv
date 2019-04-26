
# coding: utf-8

# In[1]:


import cv2
import os
import numpy as np


# In[2]:


os.getcwd(),os.chdir(r'C:\Users\14049\WordAndStudy\Projects\学校\大三上\多媒体技术\课设'),os.getcwd()


# In[3]:


def cv2ShowImages(imgs):
    for i,img in enumerate(imgs):
        cv2.namedWindow(str(i),cv2.WINDOW_NORMAL)
        cv2.imshow(str(i),img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# # VideoCapture?
# ```python
# import numpy as np
# import cv2
# 
# cap = cv2.VideoCapture('vtest.avi')
# 
# while(cap.isOpened()):
#     ret, frame = cap.read()
# 
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# 
# cap.release()
# cv2.destroyAllWindows()
# ```

# In[33]:


cap = cv2.VideoCapture('video-h265.mkv')


# In[34]:


frameCount = 0
pre = None
frame = None
grayPre = None
grayCur = None

while(cap.isOpened()):
    ret,frame = cap.read()
    
    if not ret:
        break
    
    # 初始化
    if frameCount ==0:
        grayPre = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    grayCur = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # diff
    diff = cv2.absdiff(grayPre,grayCur)
    _,thresDiff = cv2.threshold(diff,2,255,cv2.THRESH_BINARY)
    
    erodeKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18));
    # dilateKernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (18, 18));
    
    tmp = cv2.erode(thresDiff,erodeKernel)
    # tmp = cv2.dilate(tmp,dilateKernel)
    
    
    cv2.imshow("frame",frame)
    # cv2.imshow("diff",diff) # diff 值太小不明显
    cv2.imshow("thresdiff",thresDiff)
    cv2.imshow("Kernel",Kernel)
    cv2.imshow('tmp',tmp)
    
    # 更替
    grayPre = grayCur.copy()
    frameCount +=1
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


# In[9]:


get_ipython().run_line_magic('pinfo', 'cv2.threshold')


# In[25]:


cap.release()


# In[26]:


cv2.destroyAllWindows()


# # 积累
