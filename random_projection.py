import numpy as np
import math
from matplotlib import pyplot as plt

d = 1000
k=[10, 30, 50, 80, 100, 150, 200, 300, 400, 500, 600, 800, 1000]

def randomProjection(X, k):
    A = np.random.normal(1/k, 1/math.sqrt(k),(k,d))
    return (1/math.sqrt(k))*(np.dot(A,X))

X = np.random.randint(-100,100,(d,d))
AxNorm = []
for i in k:
    AxNorm.append(np.linalg.norm(randomProjection(X,i)))

xNorm = np.linalg.norm(X)
y=[i/xNorm for i in AxNorm]
print(y)
print(xNorm)
print(AxNorm)

plt.plot(k,y,'ro')
plt.title("D=10000")
plt.xlabel("K values")
plt.ylabel("AxNorm/xNorm")
plt.show()
