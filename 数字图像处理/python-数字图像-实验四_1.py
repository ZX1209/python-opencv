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


# # 彩色图像的颜色通道的值提高
# > 注意 np 数组溢出
# In[6]:
cImg = cv2.imread(r'baboon.bmp')


# In[7]:
cImgList2 = cImg.tolist()


# In[8]:
tmpImg = cImg.astype('int32')+30


# In[9]:
tmpImg = tmpImg.clip(0, 255)


# In[10]:
tmpImg = tmpImg.astype('uint8')


# In[11]:
rl = len(cImgList2)
cl = len(cImgList2[0])

bias = 30

for ri in range(rl):
    for ci in range(cl):
        for i in range(3):
            cImgList2[ri][ci][i] = max(0, min(cImgList2[ri][ci][i]+bias, 255))


# In[12]:
tmp1 = np.array(cImgList2, 'uint8')


# In[13]:
cv2ShowImages([cImg, np.array(cImgList2, 'uint8'), tmpImg])


# In[14]:
ans = tmp1 == tmpImg


# In[15]:
CImg = cImg+30
CImg = CImg.clip(1, 255)
cv2ShowImages([cImg, CImg.astype('uint8')])


# # 图像去饱和度
# In[16]:
cImgList = cImg.tolist()
dImg = cImgList.copy()


# In[17]:
alpha_silder = 50
alpha_slider_max = 100


# In[18]:
rl = len(cImgList)
cl = len(cImgList[0])

alpha = alpha_silder/alpha_slider_max
beta = 1-alpha
for ri in range(rl):
    for ci in range(cl):
        b, g, r = cImgList[ri][ci]
        y = 0.114*b+0.587*g+0.299*r
        dImg[ri][ci] = [alpha*y+beta*b, alpha*y+beta*g, alpha*y+beta*r]


# In[19]:
cv2ShowImages([cImg, np.array(dImg, 'uint8')])

# # HSV 各分量调整
# In[21]:
hsvImg = cv2.cvtColor(cImg, cv2.COLOR_BGR2HSV)


# In[22]:
# h,s,v = cv2.split(hsvImg)
# cv2ShowImages([h,s,v])
# In[25]:
# 有些许问题呢
cv2.namedWindow('image')

alpha = 0
alpha_silder = 0

alpha_slider_max = 100
alpha_slider_min = 0


def doNothing(no):
    pass


# 创建滑块,注册回调函数
cv2.createTrackbar('h', 'image', alpha_slider_min, alpha_slider_max, doNothing)
cv2.createTrackbar('s', 'image', alpha_slider_min, alpha_slider_max, doNothing)
cv2.createTrackbar('v', 'image', alpha_slider_min, alpha_slider_max, doNothing)

while True:
    k = cv2.waitKey(1) & 0xFF
    if k == 8:
        break
    hValue = cv2.getTrackbarPos('h', 'image')
    sValue = cv2.getTrackbarPos('s', 'image')
    vValue = cv2.getTrackbarPos('v', 'image')

    h, s, v = cv2.split(hsvImg)
    h = h/180
    s = s/255
    v = v/255

    h = np.clip(h+(hValue-50)/50, 0, 1.0)
    h *= 180
    h = h.astype('uint8')

    s = np.clip(s+(sValue-50)/50, 0, 1.0)
    s *= 255
    s = s.astype('uint8')

    v = np.clip(v+(vValue-50)/50, 0, 1.0)
    v *= 255
    v = v.astype('uint8')

    dImg = cv2.merge((h, s, v))
    dImg = cv2.cvtColor(dImg, cv2.COLOR_HSV2BGR)

    cv2.imshow('image', dImg)

cv2.destroyAllWindows()
