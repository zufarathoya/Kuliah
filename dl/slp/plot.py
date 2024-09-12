import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

#matplotlib.use('TkAgg')

data = pd.read_csv('result.csv')

plt.plot(data['x'].iloc[0:10], data['y'].iloc[0:10], label='Train')
plt.plot(data['x'].iloc[20:30], data['y'].iloc[20:30], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy plot')
plt.legend()
plt.show()

plt.plot(data['x'].iloc[10:20], data['y'].iloc[10:20], label='Train')
plt.plot(data['x'].iloc[30:40], data['y'].iloc[30:40], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Error')
plt.title('Error plot')
plt.legend()
plt.show()
