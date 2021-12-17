import matplotlib.pyplot as plt

image = plt.imread('../data/classify-leaves/images/0.jpg')

plt.figure(1)
plt.imshow(image)
plt.figure(2)
x=[0,1,2,3,4,5,6,7,8,9,10]
plt.plot(x)
# plt.show()

import pandas as pd 
df = pd.read_csv("..\\data\\classify-leaves\\train.csv")
print(df.shape)