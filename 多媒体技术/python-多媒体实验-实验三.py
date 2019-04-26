
# coding: utf-8

# In[9]:


import matplotlib.pyplot as plt
import numpy as np


# In[10]:


x = np.linspace(0,1,30)


# In[11]:


f = 1/(1+x)


# In[12]:


g = np.e**x


# In[27]:


# %matplotlib
fl, = plt.plot(x,f,'r*',label='f func')
gl, = plt.plot(x,g,'b*',label = 'g func')
plt.legend(handles=[fl,gl])
plt.show()

