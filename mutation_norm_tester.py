import matplotlib.pyplot as plt 
import numpy as np 
from copy import copy 
P = np.asarray([0.4,0.1,0.05,0.45])
samples = 100000
record = []
record2 = []
for i in range(samples):
    
    P_prime = copy(P)
    for j in range(len(P)):
        X = np.random.uniform(0.95,1.05)
        P_prime[j] *= X 
        if P_prime[j] > 0.95 :
            P_prime[j] = 0.95
        if P_prime[j] < 0.05 :
            P_prime[j] = 0.05
    P_prime /= sum(P_prime)
    record += [P_prime[0]-P[0]]
    P = P_prime
    record2 += [P[0]]
print("Avg: {0}, Avg Abs: {1}".format(sum(record)/len(record), sum([abs(x) for x in record])/len(record)))
plt.hist(record)
plt.show()
plt.plot(record2)
plt.show()