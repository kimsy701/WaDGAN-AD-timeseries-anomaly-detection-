#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt


# In[29]:


#비교 data A
T=4*24*365*3
x = np.arange(0,T)
cycle_param = (2*np.pi)/(4*24*365)
y = np.sin(cycle_param*x) + 1 #0이상의 값


# In[10]:


plt.plot(y)


# In[40]:


noise = 0.3 *np.random.normal(0, 1, 105120) 


# In[41]:


plt.plot(y+noise)


# In[37]:


#비교 data B
T = 4*24*365*3
x = np.arange(0,T)
cycle_param1 = (2*np.pi)/(4*24*365) #1년 짜리 주기
cycle_param2 =(2*np.pi)/(4*24*7)  #일주일 짜리 주기
y = np.sin(cycle_param1*x) + np.sin(cycle_param2 * x) +2 #0방지


# In[22]:


plt.plot(y)


# In[39]:


plt.plot(y[0:4*24*7]


# In[24]:


#비교 data C
T = 4*24*365*3
x = np.arange(0,T)
cycle_param1 = (2*np.pi)/(4*24*365) #1년 짜리 주기
cycle_param2 =(2*np.pi)/(4*24*7)  #일주일 짜리 주기
cycle_param3 = (2*np.pi)/(4*24) #하루짜리 주기
y = np.sin(cycle_param1*x) + np.sin(cycle_param2 * x) + np.sin(cycle_param3 * x) +3


# In[25]:


plt.plot(y[0:4*24*7])


# In[28]:





# In[ ]:




