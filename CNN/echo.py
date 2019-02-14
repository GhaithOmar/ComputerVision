import matplotlib.pyplot as plt 
import numpy as np 
import wave # let us work with wave files
import sys 

from scipy.io.wavfile import write

spf = wave.open('helloworld.wav', 'r')
# in order to turn the signal into numpy array  we need to call the
# function readframes and then we need to call a function
# fromstring the second parameter should specifying that it's encoded
# into 16 bit integer
signal = spf.readframes(-1)
signal = np.fromstring(signal, 'Int16')

print(f'numpy singla shape {signal.shape}')

plt.plot(signal)
plt.title('Hello world without echo')
plt.show()

delta = np.array([1.,0., 0.])
noecho = np.convolve(signal, delta)
print('noecho signal:', noecho.shape)
assert(np.abs(noecho[:len(signal)] - signal).sum() < 0.0000001)

noecho = noecho.astype(np.int16)
write('noecho.wav', 16000, noecho)

filt = np.zeros(16000)
filt[0] = 1
filt[4000] = 0.6 
filt[8000] = 0.3
filt[12000] = 0.2
filt[15999] = 0.1 
out =np.convolve(signal, filt)

out = out.astype(np.int16)
write('out.wav', 16000, out)