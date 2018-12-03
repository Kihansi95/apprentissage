from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')  

print(mnist) 
print (mnist.data) 
print (mnist.target) 
len(mnist.data) 
help(len)     
print (mnist.data.shape) 
print (mnist.target.shape) 
mnist.data[0] 
mnist.data[0][1] 
mnist.data[:,1] 
mnist.data[:100] 
