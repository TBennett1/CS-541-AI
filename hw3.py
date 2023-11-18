import numpy as np
import matplotlib.pyplot as plt

n=100
d=200

X=np.random.randn(n,d)
y=np.random.randn(n)

wStar= np.matmul(np.linalg.matrix_power((np.matmul(np.transpose(X),X)),-1),np.matmul(np.transpose(X),y))
print(wStar)
hessian = np.matmul(np.transpose(X),X)

eig = np.linalg.eigvals(hessian)
L = 1/max(eig)
eta = [0.01*L,0.1*L,L,2*L,20*L,100*L]

results=[[],[],[],[],[],[]]
for i in range(0,6):
    cur_w = [0]*d
    for t in range(1,101):
        cur_w = np.subtract(cur_w, eta[i]*(2*(np.subtract(np.matmul(np.matmul(np.transpose(X),X), cur_w), np.matmul(np.transpose(X),y)))))
        results[i].append(np.linalg.norm(np.subtract(y,np.matmul(X,cur_w))))

t=list(range(1,101))
for i in range(0,6):
    j = eta[i]
    plt.plot(t,results[i])
    plt.xlabel('t')
    plt.show()