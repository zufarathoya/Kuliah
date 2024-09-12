import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv('result.csv')

plt.figure(figsize=(12, 6))
plt.plot(data['epoch'].iloc[0:10], data['accuracy'].iloc[0:10], label='Train')
plt.plot(data['epoch'].iloc[10:20], data['accuracy'].iloc[10:20], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Plot')
plt.legend()
plt.show()

plt.figure(figsize=(12, 6))
plt.plot(data['epoch'].iloc[0:10], data['meanError'].iloc[0:10], label='Train')
plt.plot(data['epoch'].iloc[10:20], data['meanError'].iloc[10:20], label='Test')
plt.xlabel('Epoch')
plt.ylabel('Mean Error')
plt.title('Error Plot')
plt.legend()
plt.show()
