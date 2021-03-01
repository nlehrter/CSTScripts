import regex as re
import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import control

class CST_SParams:
    
    def __init__(self, filepath, x = None,y= None):
        if(filepath !=None):
            self.x_data, self.y_data = self.read_data(filepath)
        else:
            self.x_data, self.y_data = x,y
        self.x_delay, self.y_delay = self.window_and_transform(self.x_data,self.y_data)
        
    def read_data(self,filename):
        with open(filename) as file:
            data = file.readlines()[2:]
        x = []
        y = []
        for i in range(0,len(data)-1):
            
            holding = re.findall("-*\d.{0,1}\d*",data[i])
            x.append(float(holding[0]))
            y.append(float(holding[1]))
        if (np.mean(y[:1000]) < 0):
            y = list(map(control.db2mag,y))
        return x,y

    def window_and_transform(self,x,y):
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
        
        N = len(x)
        y = np.array(y)
        x = np.array(x)
    
        T = 1/(x[1]-x[0])    
        
        xT = sp.fft.fftfreq(N,1/T)[:N//2]
        xT = [xT[l]* 1e9 for l in range(0,len(xT))]
        delay_window = sp.fft.ifft(y)[:N//2]
        
        return xT[:N//2],abs(delay_window.real)[:N//2]
    
    def zero_pad(self, scale_factor, writeback = False):
        x = self.x_data
        y = self.y_data
        x.extend([(x[i]-x[0] + x[len(x)-1]) for i in range(0,int(len(x)*(scale_factor-1)))])
        y.extend([0 for i in range(0,int(len(y)*(scale_factor-1)))])
        
        if (writeback == True):
            self.__init__(None,x,y)

