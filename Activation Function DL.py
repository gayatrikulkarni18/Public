#!/usr/bin/env python
# coding: utf-8

# In[15]:


#Sigmoid Function
import matplotlib.pyplot as plt
import numpy as np
def sigmoid(x):
    s=1/(1+np.exp(-x))
    ds=s*(1-s)
    return s,ds
x=np.arange(-6,6,0.01)
sigmoid(x)
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.plot(x,sigmoid(x)[0],color='#307EC7',linewidth=3, label="sigmoid")
ax.plot(x,sigmoid(x)[1],color='#9621e2',linewidth=3, label="derivative")
fig.show()
print("Gayatri Kulkarni - 53004230002")


# In[16]:


#Tanh Function
import matplotlib.pyplot as plt
import numpy as np
def tanh(x):
    t=(np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    dt=1-t**2
    return t,dt
z=np.arange(-4,4,0.01)
tanh(z)[0].size,tanh(z)[1].size
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.plot(x,sigmoid(x)[0],color='#307EC7',linewidth=3, label="tanh")
ax.plot(x,sigmoid(x)[1],color='#9621e2',linewidth=3, label="derivative")
ax.legend(loc="upper right", frameon=False)
plt.show()
print("Gayatri Kulkarni - 53004230002")


# In[17]:


#Relu Function and its derivative
import matplotlib.pyplot as plt
import numpy as np
def relu(x):
    r = np.maximum(0, x)
    dr = np.where(x <=0, 0, 1)
    return r,dr

x=np.arange(-6, 6, 0.1)#Range for the x-axis
relu_values, relu_derivatives = relu(x) #Compute ReLU AND its derivative
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position(('data', 0))#set the x-axis at y=0
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.plot(x,relu_values,color='#307EC7',linewidth=3, label="rELU")
ax.plot(x,relu_derivatives,color='#9621e2',linewidth=3, label="derivative")
ax.legend(loc="upper right", frameon=False)
plt.show()
print("Gayatri Kulkarni - 53004230002")


# In[18]:


# lEAKY Rectified Linear Unit(Leaky Relu)
import matplotlib.pyplot as plt
import numpy as np
def leaky_relu(x, alpha=0.01):
    r = np.maximum(alpha * x, x)
    dr = np.where(x < 0, alpha, 1)
    return r,dr
#Generate x values
x=np.arange(-6, 6, 0.1)#Range for the x-axis
leaky_relu_values, leaky_relu_derivatives = leaky_relu(x)

# Create the plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position(('data', 0))#set the x-axis at y=0
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
ax.plot(x, leaky_relu_values, color='#307EC7',linewidth=3, label="rELU")
ax.plot(x,leaky_relu_derivatives, color='#9621e2',linewidth=3, label="derivative")
ax.legend(loc="upper right", frameon=False)
#Show the plot
plt.show()
print("Gayatri Kulkarni - 53004230002")


# In[19]:


# Parametric Rectified Linear Unit (PRelu)
import matplotlib.pyplot as plt
import numpy as np
def prelu(x, alpha=0.25):
    r = np.maximum(alpha * x, x)
    dr = np.where(x < 0, alpha, 1)
    return r,dr

x=np.arange(-6, 6, 0.1)#Range for the x-axis
prelu_values, prelu_derivatives = prelu(x)

# Create the plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position(('data', 0))#set the x-axis at y=0
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

#Plotting the PRelu and its derivative
ax.plot(x, prelu_values, color='#307EC7',linewidth=3, label="PReLU")
ax.plot(x,prelu_derivatives, color='#9621e2',linewidth=3, label="Derivative")
ax.legend(loc="upper right", frameon=False)
#Show the plot
plt.show()
print("Gayatri Kulkarni - 53004230002")


# In[26]:


# Exponential Linear Unit (ELU)
import matplotlib.pyplot as plt
import numpy as np
def elu(x, alpha=1.0):
    r = np.where(x >= 0, x, alpha * (np.exp(x) - 1))
    dr = np.where(x <= 0, 1, alpha * np.exp(x))
    return r,dr

x=np.arange(-6, 6, 0.1)#Range for the x-axis
elu_values, elu_derivatives = elu(x)

# Create the plot
fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position(('data', 0))#set the x-axis at y=0
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

#Plotting the PRelu and its derivative
ax.plot(x, elu_values, color='#307EC7',linewidth=3, label="ELU")
ax.plot(x,elu_derivatives, color='#9621e2',linewidth=3, label="Derivative")
ax.legend(loc="upper left", frameon=False)
#Show the plot
plt.show()
print("Gayatri Kulkarni - 53004230002")


# In[21]:


import matplotlib.pyplot as plt
import numpy as np
def softplus(x):
    r = np.log(1 + np.exp(x))
    dr = 1/(1 + np.exp(x))
    return r,dr

x=np.arange(-6, 6, 0.1)#Range for the x-axis
softplus_values, softplus_derivatives = softplus(x)

fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position(('data', 0))#set the x-axis at y=0
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

#Plotting the PRelu and its derivative
ax.plot(x, softplus_values, color='#307EC7',linewidth=3, label="Softplus")
ax.plot(x,softplus_derivatives, color='#9621e2',linewidth=3, label="Derivative")
ax.legend(loc="upper left", frameon=False)
#Show the plot
plt.show()
print("Gayatri Kulkarni - 53004230002")


# In[23]:


import matplotlib.pyplot as plt
import numpy as np

def arctan(x):
    r = np.arctan(x)
    dr = 1/(1 +  x**2)
    return r,dr

x=np.arange(-6, 6, 0.1)#Range for the x-axis
arctan_values, arctan_derivatives = arctan(x)

fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position(('data', 0))#set the x-axis at y=0
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

#Plotting the PRelu and its derivative
ax.plot(x, arctan_values, color='#307EC7',linewidth=3, label="Arctan")
ax.plot(x, arctan_derivatives, color='#9621e2',linewidth=3, label="Derivative")
ax.legend(loc="upper left", frameon=False)
#Show the plot
plt.show()
print("Gayatri Kulkarni - 53004230002")


# In[25]:


import matplotlib.pyplot as plt
import numpy as np

def tanh(x):
    r = np.tanh(x)
    dr = 1- r**2
    return r,dr

x=np.arange(-6, 6, 0.1)#Range for the x-axis
tanh_values, tanh_derivatives = tanh(x)

fig, ax = plt.subplots(figsize=(9, 5))
ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position(('data', 0))#set the x-axis at y=0
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

#Plotting the PRelu and its derivative
ax.plot(x, tanh_values, color='#307EC7',linewidth=3, label="Tanh")
ax.plot(x, tanh_derivatives, color='#9621e2',linewidth=3, label="Derivative")
ax.legend(loc="upper left", frameon=False)
#Show the plot
plt.show()
print("Gayatri Kulkarni - 53004230002")


# In[ ]:




