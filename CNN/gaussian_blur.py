import numpy as np 
from scipy.signal import convolve2d 
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg


img = mpimg.imread('lena.png')

plt.imshow(img)
plt.show()

bw = img.mean(axis=2) # take the avg in the second axis

plt.imshow(bw, cmap='gray')
plt.show()


# Creating gaussian filter

W = np.zeros((20,20)) # u can but it to any size

for i in range(20):
	for j in range(20):
		dist = (i - 9.5) ** 2 + (j - 9.5) ** 2
		W[i, j] = np.exp(-dist / 50)



plt.imshow(W, cmap='gray')
plt.show()

out = convolve2d(bw, W)

plt.imshow(out, cmap='gray')
plt.show()

print(out.shape)
out = convolve2d(bw, W, mode='same')

plt.imshow(out, cmap='gray')
plt.show()

print(out.shape)

out3 = np.zeros(img.shape)

W /= W.sum()
for i in range(3):
	out3[:,:, i] = convolve2d(img[:,:,i], W, mode='same')


plt.imshow(out3)
plt.show()