import ssl
import numpy as np
import matplotlib.pyplot as plt

# parameters
def kernel(S,A):
    p = np.random.rand(S,A,S)
    for s in range(S):
        for a in range(A): 
            summ = np.sum(p[s,a])
            p[s,a] = p[s,a]/summ
    return p
   
S = 100
A = 20
R = np.random.randn(S,A)
v0 = np.random.randn(S)
gamma = 0.9
alphaSA = 0.1*np.ones((S,A))
betaSA = 0.1*np.ones((S,A))
alphaS = 0.1*np.ones(S)
betaS = 0.1*np.ones(S)
n =100
P = kernel(S,A)

class pm:
    def __init__(self,S,A):
        self.S =S 
        self.A = A
        self.R = np.random.randn(S,A)
        self.v0 = np.random.randn(S)
        self.gamma = 0.9
        self.alphaSA = 0.1*np.ones((S,A))
        self.betaSA = 0.1*np.ones((S,A))
        self.alphaS = 0.1*np.ones(S)
        self.betaS = 0.1*np.ones(S)
        self.n =100
        self.P = kernel(S,A)

