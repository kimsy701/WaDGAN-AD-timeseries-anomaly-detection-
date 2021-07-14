import pandas as pd
import matplotlib.pyplot as plt

nokji = pd.read_csv('./data/녹지캠_Dataset_RF.csv')

plt.plot(nokji.UsedPower)
plt.xlim([0,4*24*365*4])
plt.title('nokji_4years')
plt.savefig('./data/nokji_years.png')
plt.clf()
