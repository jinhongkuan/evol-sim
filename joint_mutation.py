import numpy as np 
import matplotlib.pyplot as plt 
import math 

# P(A=k) that is generated using normal dist and P(BCD=1/k) is equivalent 
# P(A=k) = (Gauss(A=k) * Gauss(BCD=k))/(Gauss(A=j) * Gauss(BCD=j) for all j)
# Or maybe lognormal distribution?

data = np.random.lognormal(mean=0, sigma=0.3,size=(10000))
data2 = np.random.lognormal(mean=0, sigma=0.3,size=(10000))
data3 = np.random.lognormal(mean=0, sigma=0.3,size=(10000))
data4 = np.random.lognormal(mean=0, sigma=0.3,size=(10000))
data_inverse = [a/b for a,b,c,d in zip(data,data2,data3,data4)]
print(data)
plt.hist(data,bins=100)
plt.show()
print(data_inverse)
plt.hist(data_inverse,bins=100)
plt.show()