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

test = CST_SParams("../data/S21_Coax2.txt")

fig = plt.figure(2)
plt.semilogy(test.x_delay, test.y_delay, color = 'red', label = '12m Coax')
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

test.zero_pad(1.5,True)
plt.semilogy(test.x_delay, test.y_delay, color = 'blue', label = 'Zero Padding')