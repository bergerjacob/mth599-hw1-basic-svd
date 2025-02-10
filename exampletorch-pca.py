import torch
import time
import matplotlib.pyplot as plt


X = torch.load('benny_v2.pt',weights_only=True)

def show_frame(image,k):
    plt.imshow(image[k,:,:,:].permute(1,2,0).numpy())
    plt.show()

print(X.shape)
#show_frame(X,0)

#X = X[:2000]

sz = X.shape[1:]
X = X.reshape(X.shape[0],-1)

print(X.shape)

t0 = time.time()
U,S,V = torch.pca_lowrank(X, q=50, center=True, niter=2)
t1 = time.time()
dt = t1 -t0
print("time=",dt)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

x = U[:,0].numpy()
y = U[:,1].numpy()
z = U[:,2].numpy()

# Plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z)

plt.figure()
plt.scatter(x, y)

plt.figure()
v1 = V[:,0]
v1 = (v1 - v1.min())/(v1.max()-v1.min())
plt.imshow(v1.reshape(sz).permute(1,2,0).numpy())

plt.figure()
v2 = V[:,1]
v2 = (v2 - v2.min())/(v2.max()-v2.min())
plt.imshow(v2.reshape(sz).permute(1,2,0).numpy())

plt.figure()
v3 = V[:,2]
v3 = (v3 - v3.min())/(v3.max()-v3.min())
plt.imshow(v3.reshape(sz).permute(1,2,0).numpy())




plt.show()








