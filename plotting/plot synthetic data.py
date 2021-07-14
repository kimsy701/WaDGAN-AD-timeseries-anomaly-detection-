import pandas as pd
import matplotlib.pyplot as plt


dataA = pd.read_csv('dataA.csv').iloc[:70080,1]
dataB = pd.read_csv('dataB.csv').iloc[:70080,1]
dataC = pd.read_csv('dataC.csv').iloc[:70080,1]

plt.figure(figsize=(40,10))
plt.plot(dataA)
plt.plot(dataB)
plt.plot(dataC)
plt.savefig('syntheticdata11.png')

nokji = pd.read_csv('녹지캠_Dataset_RF.csv')

