import codecademylib3_seaborn
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
digits = datasets.load_digits()
print(digits)
print(digits.DESCR)
print(digits.data)
print(digits.target)
plt.gray()
plt.matshow(digits.images[100])
plt.show()
print(digits.target[100])
model = KMeans(n_clusters=10, random_state=42)
model.fit(digits.data)
fig = plt.figure(figsize=(8,3))
fig.suptitle('Cluser Center Images', fontsize=14, fontweight='bold')
for i in range(10):

  # Initialize subplots in a grid of 2X5, at i+1th position
  ax = fig.add_subplot(2, 5, 1 + i)

  # Display images
  ax.imshow(model.cluster_centers_[i].reshape((8, 8)), cmap=plt.cm.binary)
plt.show()
new_samples = np.array([
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.76,2.29,2.29,2.29,2.06,0.00,0.00,0.00,6.17,6.10,5.34,5.34,5.11,0.00,0.00,0.00,7.17,1.52,0.00,0.00,0.00,0.00,0.00,0.31,7.63,4.27,3.05,0.99,0.00,0.00,0.00,0.76,7.62,4.27,5.49,7.40,3.73,0.00,0.00,0.23,6.00,6.10,2.21,2.44,7.62,0.69,0.00,0.00,0.38,3.66,6.71,7.47,6.79,0.38,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,2.67,5.87,1.60,0.00,0.00,0.00,0.00,0.00,5.03,7.55,4.57,0.00,0.00,0.00,0.00,0.00,0.23,4.05,4.88,0.15,0.00,0.00,0.00,1.91,5.57,7.32,7.63,4.34,0.00,0.00,0.00,1.91,3.81,2.29,4.20,4.57,0.00,0.00,0.00,3.35,5.34,6.10,7.32,3.20,0.00,0.00,0.00,1.22,2.29,2.14,0.76,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.37,6.63,6.71,6.63,5.64,1.22,0.00,0.00,3.66,7.62,5.03,1.45,5.79,4.04,0.00,0.00,2.97,7.62,4.19,0.00,3.81,4.57,0.00,0.00,0.23,5.87,4.65,0.15,4.88,4.57,0.00,0.00,0.00,2.82,7.47,7.17,6.71,1.83,0.00,0.00,0.00,0.00,0.92,1.37,0.15,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00],
[0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,1.22,0.68,0.00,0.00,0.00,0.00,0.00,0.00,5.42,3.05,0.00,0.00,0.00,0.00,0.00,0.61,7.47,1.60,0.00,0.00,0.00,0.00,0.00,2.59,6.56,0.00,0.00,0.00,0.00,0.00,0.00,3.74,5.11,0.00,0.00,0.00,0.00,0.00,0.00,4.73,4.34,0.00,0.00,0.00,0.00,0.00,0.00,2.29,1.37,0.00,0.00,0.00,0.00]
])
new_labels = model.predict(new_samples)
print(new_labels)
for i in range(len(new_labels)):
  if new_labels[i] == 0:
    print(0, end='')
  elif new_labels[i] == 1:
    print(9, end='')
  elif new_labels[i] == 2:
    print(2, end='')
  elif new_labels[i] == 3:
    print(1, end='')
  elif new_labels[i] == 4:
    print(6, end='')
  elif new_labels[i] == 5:
    print(8, end='')
  elif new_labels[i] == 6:
    print(4, end='')
  elif new_labels[i] == 7:
    print(5, end='')
  elif new_labels[i] == 8:
    print(7, end='')
  elif new_labels[i] == 9:
    print(3, end='')

