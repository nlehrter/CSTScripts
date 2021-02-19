import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import fftpack
from scipy.fft import fft, fftfreq
import statistics
import regex as re
import control



def extract_data(filename):
    with open(filename) as file:
        data = file.readlines()[2:]
    x = [0]
    y = [0]
    for i in range(1,len(data)-1):
        holding = re.findall("\S+",data[i])
        x.append(float(holding[0]))
        y.append(float(holding[1]))
    return x,y
    
def FFT(x,y):
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
    delay_vals = sp.fft.fft(yf)[:N//2]
    delay_window = sp.fft.fft(y)[:N//2]
    xs = np.linspace(0,100,len(delay_vals))
    
    return xT[:N//2],abs(delay_window.real)[:N//2]

    
    
xin,yin = extract_data("TF/input_sig.txt")

xout,yout = extract_data("TF/output_sig.txt")

xTF = []
yTF = []

index = 12
for i in range(13,len(xin)):
    match = round(xin[i], significant_digits - int(math.floor(math.log10(abs(a_number)))) - 1)
    found = False
    while (found == False):
        if (match == xout[index]):
            xTF.append(match)
            yTF.append(yout[index]/yin[i])
            found = True
        elif(index == len(xout)-1):
            break
        else:
            index += 1
rounded_number =   
        
"""
N2 = len(xin)
w2 = sp.signal.blackmanharris(N2)
yin = yin*w2
N = len(xout)
w = sp.signal.blackmanharris(N)
yout = yout*w
#plt.plot(xin,yin, color = 'cyan')
#plt.plot(xout,yout, color = 'blue')

fxin,fyin = FFT(xin,yin)
fxout,fyout = FFT(xout,yout)
plt.semilogy(fxin,fyin, label= 'Input')
plt.semilogy(fxout,fyout,  label = 'Output')
plt.legend()
print(len(fxin)-len(fxout))

    
"""
"""
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
"""