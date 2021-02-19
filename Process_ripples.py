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

def Transform_cst_data(filename):
    with open(filename) as file:
        data = file.readlines()[2:]
    
    x = []
    y = []
    for i in range(0,len(data)-1):
        holding = re.findall("-*\d.{0,1}\d*",data[i])
        x.append(float(holding[0]))
        y.append(float(holding[1]))
        
    print(abs(x[5]-x[6]))
    N2 = len(x)
    w2 = sp.signal.blackmanharris(N2)
    y = y*w2
    
    x2 = x.copy()[1:]
    y2 = y.copy()[1:]
    y2 = np.flip(y2)
    x2.reverse()
    x3 = [x2[j]*-1 for j in range(0,len(x2))]
    y = np.append(y,y2) 
    x =  x + x3 
    
    x = [x[j]*1e9 for j in range(0,len(x))]
    yf = [y[k] for k in range(0,len(y))]
    
    
    N = len(x)
    y = np.array(y)
    x = np.array(x)
    T = 1/(x[1]-x[0])
    print(T*N)
    print("The sampling rate of this signal is: {}".format(1/(x[1]-x[0])))
    
    
    xT = sp.fft.fftfreq(N,1/T)[:N//2]
    xT = [xT[l]* 1e9 for l in range(0,len(xT))]
    delay_vals = sp.fft.ifft(yf)[:N//2]
    delay_window = sp.fft.ifft(y)[:N//2]
    xs = np.linspace(0,100,len(delay_vals))
    
    return xT[:N//2],abs(delay_window.real)[:N//2]

    
delay_standard, magnitude_standard = Transform_cst_data("../data/S21_Coax_variations.txt")
delay_cracks, magnitude_cracks = Transform_cst_data("../data/S21_Coax_cracks.txt")
S21_12m = CST_SParams("../data/S21_Coax_long.txt")

fig = plt.figure(2)
plt.semilogy(S21_12m.x_delay, S21_12m.y_delay, color = 'red', label = 'Coax with variations')
plt.title('Projected Delay Spectrum of $S_{21}$ for 10m cable with cracks', fontsize = 32)
plt.semilogy(delay_cracks, magnitude_cracks, 'b',label = 'Model with inserted cracks')
plt.xlabel('Delay (ns)', fontsize = 40)
plt.ylabel('Amplitude', fontsize = 40)
plt.xlim(0,400)
plt.ylim(1e-11,0.9)
plt.xticks([0,50,100,150,200,250,300,350,400])
plt.legend(fontsize = 35)
ax = plt.gca()
ax.tick_params(axis='both', which='major', labelsize=26)
plt.grid()