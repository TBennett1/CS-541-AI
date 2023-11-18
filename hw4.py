import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from collections import Counter
import math

n=610
p=193609

M = np.zeros((n,p))

omega = []

with open("/Users/tbennett/Documents/School/CS 541/HW4/ratings.csv", newline='') as csvfile:
    reader = csv.DictReader(csvfile, fieldnames=["userId","movieId","rating","timestamp"])
    next(reader)

    for entry in reader:
        M[int(entry['userId'])-1][int(entry['movieId'])-1] = float(entry["rating"])
        omega.append((int(entry['userId'])-1,int(entry["movieId"])-1))


np.random.shuffle(M)
def Diff(a,b):
    ca = Counter(a)
    cb = Counter(b)
    return (ca-cb).elements()
omega1 = random.choices(omega,k=549)
omega2 = Diff(omega,omega1)

r=np.linalg.matrix_rank(M)

U = np.random.rand(n,r)
V = np.random.rand(p,r)

currU = U
currV = V

results = []
it = 0
for (i,j) in omega1:
    currU = currU * 0.01 * (M[i][j]-currU[i]*np.transpose(V[j])*(-V[j])+currU[i])
    currV = currV * 0.01 * (M[i][j]-currV[i]*np.transpose(V[j])*(-V[j])+currV[j])
    results.append((1/2)*sum((M[i][j]-currU[i]*np.transpose(currV[j]))**2 + (1/2)*(np.linalg.norm(currU,'fro')**2 + np.linalg.norm(currV,'fro')**2)))
    it+=1

plt.plot(list(range(it)),results)
plt.show()


X= currU*np.transpose(currV)
RMSE = math.sqrt((1/np.linalg.norm(omega2,1))*((M-X)**2))

currU = U
currV = V
lam = [10**-6,10**-3,0.1,0.5,2,5,10,20,50,100,500,1000]
it=0
RSMElst = []
for l in lam:
    results=[]
    for (i,j) in omega2:
        currU = currU * 0.01 * (M[i][j]-currU[i]*np.transpose(V[j])*(-V[j])+(lam*currU[i]))
        currV = currV * 0.01 * (M[i][j]-currV[i]*np.transpose(V[j])*(-V[j])+(lam*currV[j]))
        results.append((1/2)*sum((M[i][j]-currU[i]*np.transpose(currV[j]))**2 + (1/2)*(np.linalg.norm(currU,'fro')**2 + np.linalg.norm(currV,'fro')**2)))
        it+=1
    X= currU*np.transpose(currV)
    RMSE = math.sqrt((1/np.linalg.norm(omega2,1))*((M-X)**2))
    RSMElst.append(RMSE)

plt.plot(lam,RSMElst)
plt.show()
