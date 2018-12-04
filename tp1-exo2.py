from sklearn import datasets 
import matplotlib.pyplot as plt 
mnist = datasets.fetch_mldata('MNIST original') 
images = mnist.data.reshape((-1, 28, 28)) 
plt.imshow(images[0],cmap=plt.cm.gray_r,interpolation="nearest") 
plt.show() 

plt.imshow(data.reshape((-1, 28, 28))[0],cmap=plt.cm.gray_r,interpolation="nearest")
