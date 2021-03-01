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


def ACF_S21(data1,data2):
    
    x1,y1 = data1[0],data1[1]
    x2,y2 = data2[0],data2[1]
    
    if (x1 == x2):
        y = [a*b for a,b in zip(y1,y2)]
        ACF = CST_SParams(None,x1,y)
        return ACF
    return None    

#delay, magnitude = Transform_cst_data("../S21_Coax_variations.txt")

S21_10m = CST_SParams("../data/coax_10m.txt")
S21_12m = CST_SParams("../data/coax_12m.txt")

print(type(S21_10m.x_data))
cross_correl = ACF_S21((S21_12m.x_data, S21_12m.y_data),(S21_10m.x_data,S21_10m.y_data))

delay_standard, magnitude_standard = cross_correl.x_delay, cross_correl.y_delay
fig = plt.figure(2)
plt.semilogy(delay_standard, magnitude_standard, color = 'red', label = 'Coax CCF')
plt.semilogy(S21_12m.x_delay, S21_12m.y_delay, color = 'blue', label = '12m Coax')
plt.semilogy(S21_10m.x_delay, S21_10m.y_delay, color = 'green', label = '10m Coax')
#plt.semilogy(delay, magnitude, color = 'blue', label = 'Coax delay spectrum')
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