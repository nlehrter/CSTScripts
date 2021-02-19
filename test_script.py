import pandas as pd
import regex as re

with open("../ripple_data.txt") as file:
    data = file.readlines()[2:]
    
    x = []
    y = []
    for i in range(0,len(data)-1):
        holding = re.findall("-*\d.{0,1}\d*",data[i])
        x.append(float(holding[0]))
        y.append(float(holding[1]))