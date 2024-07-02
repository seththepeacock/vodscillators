import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import welch

T = 100
f0 = 4096
N = T*f0

np.random.seed(43)

intervalle=np.linspace(0,T,T*f0)
bruit = np.random.normal(size=(T*f0))

#plt.figure()
#plt.plot(intervalle, bruit)
#plt.show()

dtildaf = np.fft.fft(bruit)
freq = np.fft.fftfreq(T*f0,1/f0)
PE = np.loadtxt('PE.txt')
print(len(freq))
"""
plt.figure()
plt.plot(freq, amp2)
plt.show()
"""

Pi = welch(bruit, fs=f0, nperseg=2*f0, average='median')
'''
plt.figure()
plt.plot(Pi[0],Pi[1])
plt.title('welch')
plt.show()'''
###############
freq=freq[100:(len(freq)) //2+1]
for i in range(T):
	dtildaf[i]=0
##############
PE_int = np.interp(freq, PE[:,0], PE[:,1] )
Pi_int = np.interp(freq, Pi[0], Pi[1])
PE_int=np.concatenate((np.zeros(100),PE_int))
PE_int=np.concatenate((PE_int,PE_int[-1:0:-1]))
PE_int=np.delete(PE_int,len(freq)//2)
Pi_int=np.concatenate((np.ones(100),Pi_int))
Pi_int=np.concatenate((Pi_int,Pi_int[-1:0:-1]))
Pi_int=np.delete(Pi_int,len(freq)//2)
dftildef = PE_int*np.sqrt(1/Pi_int)*dtildaf
dff =  np.fft.ifft(dftildef)
print(dff)
#plt.figure()
#plt.plot(np.real(dff))
#plt.show()

Pf = welch(np.real(dff), fs=f0, nperseg=2*f0, average='median')

figure1 = plt.figure()
plt.plot(Pf[0],np.sqrt(Pf[1]))
plt.plot(PE[:,0], PE[:,1])
plt.xscale('log')
plt.yscale('log')
plt.show()

figure2 = plt.figure()
plt.plot(intervalle, np.real(dff))
plt.show()

'''
def fenetre(x,omega):
	if x>omega*np.pi/2 or x<-omega*np.pi/2:
		return 0
	else:
		return np.cos(x/omega)

ft = np.fft.fft(bruit*np.hanning(T*f0))
amplitude = np.absolute(ft)

amp2 = amplitude **2 * T / N / N
plt.figure()
plt.plot(freq,amp2)	
plt.title('nous')
plt.show()
'''

temporel = np.savetxt("temporel.txt", np.real(dff))
spectr = np.savetxt('spectre.txt', dftildef)
