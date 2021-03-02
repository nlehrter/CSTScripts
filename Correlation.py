import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import fftpack
from scipy.fft import fft, fftfreq
import statistics
import regex as re
import control
import seaborn as sns
from CST_Class import CST_SParams

plt.style.use('seaborn')
def ACF_S21(data1,data2):
    
    x1,y1 = data1[0],data1[1]
    x2,y2 = data2[0],data2[1]
    
    dk = (x1[2]-x1[1])
    
    one = np.array(y1)
    two = np.array(y2)
    auto = np.correlate(y1,y2, mode = "full")
    
    autocorrelation = auto[auto.size//2:]
    return autocorrelation,x1

#delay, magnitude = Transform_cst_data("../S21_Coax_variations.txt")

# Now create a test data set with a known ACF to check the function

def tst1(t, f = 100):
    return np.sin(2*np.pi*f*t)

def tst2(t,phi, f=100):
    return np.sin(2*np.pi*f*t + phi)


times = np.arange(0,100,0.0001)
output,k = ACF_S21((times,tst1(times)),(times,tst2(times,np.pi/4)))
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(times,tst1(times), label = 'Sine')
ax.plot(times,tst2(times, np.pi/4), label = 'Sine with phase offset')
ax.plot(k,output, label = 'ACF')
ax.set_xlim(0.01,0.05)
ax.legend()
"""
S21_10m = CST_SParams("../data/coax_10m.txt")
S21_12m = CST_SParams("../data/coax_12m.txt")

correlation1,k1 = ACF_S21((S21_12m.x_delay, S21_12m.y_delay),(S21_10m.x_delay,S21_10m.y_delay))

#delay_standard, magnitude_standard = cross_correl.x_delay, cross_correl.y_delay
fig = plt.figure(2)

plt.semilogy(S21_12m.x_delay, S21_12m.y_delay, color = 'blue', label = '12m Coax')
plt.semilogy(S21_10m.x_delay, S21_10m.y_delay, color = 'green', label = '10m Coax')
plt.semilogy(k1,correlation1, label = 'ACF 12m')
plt.title('Projected Delay Spectrum of $S_{21}$ for 10m cable with cracks', fontsize = 32)
plt.xlabel('Delay (ns)', fontsize = 40)
plt.ylabel('Amplitude', fontsize = 40)
plt.xlim(0,400)
plt.ylim(1e-11,0.9)
plt.xticks([0,50,100,150,200,250,300,350,400])
plt.legend(fontsize = 35)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=26)
plt.grid()
"""