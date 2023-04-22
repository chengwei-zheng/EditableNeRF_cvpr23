#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import mediapy as media


# In[ ]:


def f(x, y):
  r = np.sqrt(x**2 + y**2)
  n = r*5+1
  z = (np.abs(x)**n + np.abs(y)**n)**(1/n)
  return z

def g(x, y):
  z = 1 - np.minimum(f(x - 0.5, y), f(x + 0.5, y))
  return np.maximum(z, 0)

n = 100
x, y = np.meshgrid(np.linspace(-1.5, 1.5, 2*n), np.linspace(-1, 1, n), indexing='xy')
plt.contourf(g(x, y))


# In[ ]:



n = 1000
x, y = np.meshgrid(np.linspace(-1.5, 1.5, 2*n), np.linspace(-1, 1, n), indexing='xy')

fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"projection": "3d"})

z = g(x, y)

z0s = [0.2, 0.5, 0.8]
colors = [cm.tab10(0), cm.tab10(1), cm.tab10(2)]

ax.plot_surface(x, y, z, linewidth=0, antialiased=False, color='gray')

x, y = np.meshgrid([-1.5, 1.5], [-1, 1], indexing='xy')
xv = np.array([-1.5, -1.5, 1.5, 1.5, -1.5])
yv = np.array([-1, 1, 1, -1, -1])

for z0, color in zip(z0s, colors):
  ax.plot3D(xv, yv, z0, zorder=10, color=color)

ax.axis(False)


# In[ ]:


vis = []
for z0, color in zip(z0s, colors):
  mask = z > z0
  vis.append(1-mask[:,:,None] * (1-np.array(color[:3])[None,None,:]))

plt.figure(figsize=(8,8))
plt.imshow(np.concatenate(vis[::-1], 0))
plt.axis(False)


# In[ ]:




