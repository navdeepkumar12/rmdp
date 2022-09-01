import numpy as np
import matplotlib.pyplot as plt
import pm
from scipy.optimize import linprog, minimize,LinearConstraint

## Non robust Bellman update ### 
def nr(v,pm,p=None, tol=None, mode= None):
    Q = np.zeros((pm.S,pm.A))
    u = np.zeros(pm.S)
    for s in range(pm.S):
        for a in range(pm.A):
            Q[s,a] = pm.R[s,a] + pm.gamma*np.dot(pm.P[s,a],v)
    for s in range(pm.S):
            u[s] = np.max(Q[s])    
    return u

############ Calculated p-variance \kappa_p(v) by binary search ###########
     #####                                    #######
def f(V,x,p=2): # Helper function,     derivative of \kappa
    return np.sum([np.sign(v-x)*np.abs(v-x)**(1/(p-1)) for v in V])


def Kappa(v,tol=0.0001, p=2, mode='auto'):  #calculates kappa: input value fucntion v, tolerance tol, p 
    if p==1:
        return 0.5*np.ptp(v)
    if p==2 and (mode =='auto' or mode =='exact') :
        return np.std(v)*np.sqrt(len(v))
    if p== 'inf':
        u = np.sort(v)
        l = int(len(v)/2)
        return np.sum(u[-l:]) - np.sum(u[:l])
    
    # Else 
    q = p/(p-1)
    ### Computes p mean, omega ###
    mx = np.max(v)
    mn = np.min(v)
    x = (mx+mn)/2
    while mx-mn > tol:
        x = (mx+mn)/2
        y = f(v,x,q)
        # print(q,x,y,mn,mx)
        if y > 0:
            mn = x
        if y < 0:
            mx = x
        if y ==0:
            return np.sum((np.abs(v-x))**q)**(1/q)
    return np.sum((np.abs(v-x))**q)**(1/q)
 
### SA rectangular L_p robust Bellman update ###    calculated p variance with binary search
def sap(v,pm,p=2,tol=0.0001,mode='auto'):
    Q = np.zeros((pm.S,pm.A))
    u = np.zeros(pm.S)
    # if p ==1:
    #      q= 'inf'
    # if p > 1:
    #     q = p/(p-1)    
    kappa = Kappa(v,tol =tol/10, p=p,mode=mode)

    for s in range(pm.S):
        for a in range(pm.A):
            Q[s,a] = pm.R[s,a] - pm.alphaSA[s,a] - pm.betaSA[s,a]*pm.gamma*kappa + pm.gamma*np.dot(pm.P[s,a],v)
    for s in range(pm.S):
            u[s] = np.max(Q[s])    
    return u
       

### S rectangular L_1 robust Bellman update ###

def s1(v,pm,p=None,tol=None, mode=None):
    Q = np.zeros((pm.S,pm.A))
    u = np.zeros(pm.S)
    kappa = 0.5*np.ptp(v)

## calculation of Q-valuesp
    for s in range(pm.S):
        for a in range(pm.A):
            Q[s,a] = pm.R[s,a] + pm.gamma*np.dot(pm.P[s,a],v)

##  Action step
    Q = np.sort(Q, axis=1)        
    for s in range(pm.S):
        penalty = pm.alphaS[s] + pm.betaS[s]*pm.gamma*kappa
        x = Q[s,-1] - penalty
        for a in range(pm.A):
            if Q[s,-1-a] > x:
                x = (np.sum(Q[s,-1-a:])-penalty)/(a+1)
            else:
                u[s] = x
                break
    return u



### S rectangular L_2 robust Bellman update ###

def s2(v,pm,p=None,tol=None, mode=None):
    Q = np.zeros((pm.S,pm.A))
    u = np.zeros(pm.S)
    kappa = np.std(v)*np.sqrt(len(v))

## calculation of Q-values
    for s in range(pm.S):
        for a in range(pm.A):
            Q[s,a] = pm.R[s,a] + pm.gamma*np.dot(pm.P[s,a],v)

##  Action step
    Q = np.sort(Q, axis=1)        
    for s in range(pm.S):
        sigma = pm.alphaS[s] + pm.betaS[s]*pm.gamma*kappa
        x = Q[s,-1] - sigma
        for a in range(pm.A):
            k = a+1
            if Q[s,-k] > x:
                temp = k*sigma*sigma + (np.sum(Q[s,-k:]))**2 - k*np.sum((Q[s]*Q[s])[-k:]) 
                x = (np.sum(Q[s,-k:])-np.sqrt(temp))/k
            else:
                u[s] = x
                break
    return u
############# s rectangular L_infty robust ############

def sinf(v,pm,p='inf',tol=None, mode=None):
    Q = np.zeros((pm.S,pm.A))
    u = np.zeros(pm.S)
    kappa = Kappa(v,p='inf') 
    for s in range(pm.S):
        for a in range(pm.A):
            Q[s,a] = pm.R[s,a] - pm.alphaS[s] - pm.betaS[s]*pm.gamma*kappa + pm.gamma*np.dot(pm.P[s,a],v)
    for s in range(pm.S):
            u[s] = np.max(Q[s])    
    return u



###########  Bellman evaluation of s-rectangular by binary search #####


def fbo(Q,x,sigma,p=2): 
    return np.sum([((q-x)**p)*(np.where(q>x,1,0)) for q in Q]) - (sigma)**p

def BOsp(Q,sigma,p=2,tol=0.001):  ## Bellman Qperator for s rectangular L_p robust
    temp = np.max(Q)
    mx = temp + 0
    mn = temp -sigma 
    x = (mx + mn)/2
    while  mx-mn > tol:
        x = (mx+mn)/2
        y = fbo(Q,x,sigma,p)
        if y > 0:
            mn = x
        if y < 0:
            mx = x 
        if y == 0:
            return x
    return x 



### S rectangular L_p robust Bellman update ###   by binary search

def sp(v,pm,p=2,tol=0.0001,mode='bin'):
    Q = np.zeros((pm.S,pm.A))
    u = np.zeros(pm.S)
    kappa = Kappa(v,tol = tol/10,p=p)   # calculation of kappa 

## calculation of Q-values
    for s in range(pm.S):
        for a in range(pm.A):
            Q[s,a] = pm.R[s,a] + pm.gamma*np.dot(pm.P[s,a],v)

##  Action step
    Q = np.sort(Q, axis=1)        
    for s in range(pm.S):
        sigma = pm.alphaS[s] + pm.betaS[s]*pm.gamma*kappa
        u[s] = BOsp(Q[s],sigma=sigma, p=p, tol=tol/10)
        
    return u



####################  Sa rectangular L_inf by Linear programming ################
def kappalpinf(v,l):
    return linprog(c=v, A_eq=np.ones((1,pm.S)), b_eq=[0], bounds = (-l,l))['fun']   


def lpsainf(v,pm, p='inf',tol=None,mode=None):
    Q = np.zeros((pm.S,pm.A))
    u = np.zeros(pm.S)

    for s in range(pm.S):
        for a in range(pm.A):
            Q[s,a] = pm.R[s,a] - pm.alphaSA[s,a]+ pm.gamma*np.dot(pm.P[s,a],v)  + kappalpinf(v,pm.alphaSA[s,a])
    for s in range(pm.S):
            u[s] = np.max(Q[s])    
    return u
       

########### Sa rectangular L_1  robust by Linear Programming  ########
def kappalp1(v,l):
    c = np.block([v,-v])
    A_eq = np.ones((1,2*pm.S))
    A_eq[0][-pm.S:] = -np.ones(pm.S)
    b_eq = np.array([0])
    A_ub = np.ones((1,2*pm.S))
    b_ub = np.array([l])
    
    return linprog(c=c, A_eq= A_eq, b_eq=b_eq, A_ub = A_ub, b_ub = b_ub, bounds = (0,1))['fun']   

def lpsa1(v,pm, p=1,tol=None,mode=None):
    Q = np.zeros((pm.S,pm.A))
    u = np.zeros(pm.S)

    for s in range(pm.S):
        for a in range(pm.A):
            Q[s,a] = pm.R[s,a] - pm.alphaSA[s,a]+ pm.gamma*np.dot(pm.P[s,a],v)  + kappalp1(v,pm.alphaSA[s,a])
    for s in range(pm.S):
            u[s] = np.max(Q[s])    
    return u


########### S rectangular L_1  robust by Linear Programming  ########
def Rd(pi,pm,s):
    A_ub = np.ones((1,pm.A))
    b_ub = np.array([pm.alphaS[0]])
    return linprog(c=pi, A_ub= -A_ub, b_ub = b_ub, bounds = (-pm.alphaS[s],0))['fun']
    
def Ker(pi,v,pm,s):
    c = np.reshape(np.outer(v,pi),-1)
    c = np.block([c,-c])
    A_eq = np.zeros((pm.S,2*pm.S*pm.A))
    for i in range(pm.S):
        A_eq[i][i*pm.A:(i+1)*pm.A] =1
        A_eq[i][pm.S*pm.A+i*pm.A:pm.S*pm.A+(i+1)*pm.A] =-1
    b_eq = np.zeros(pm.S)    
    A_ub = np.ones((1,2*pm.S*pm.A))
    b_ub = np.array([pm.betaS[s]])
    return linprog(c=c,A_eq=A_eq,b_eq=b_eq, A_ub= A_ub, b_ub = b_ub, bounds = (0,1))['fun']
    
def poleval(pi,v,pm,s):
    kappa = Rd(pi,pm,s)+pm.gamma*Ker(pi,v,pm,s)
    nom = pi@pm.R[s] + pm.gamma*pi@pm.P[0]@v
    return -nom-kappa
con = LinearConstraint(A=np.ones(pm.A),lb=1, ub=1)
# y = minimize(poleval,pi,bounds=[(0,1) for i in range(pm.A)], constraints=con)['fun']

def lps1(v,pm, p=1,tol=None,mode='lp'):
    u = np.zeros(pm.S)
    pi = np.random.rand(pm.A)
    pi = pi/np.sum(pi)
    for s in range(1):   ### only one state update , due to time constraint
        def fun(pi):
            return poleval(pi,v,pm,s)
        u[s] = -minimize(fun,pi,bounds=[(0,1) for i in range(pm.A)], constraints=con)['fun']
    return u




####################################################################################################
    


### robust value iteration n###
def setup(p=2, rect = 'sa', mode='bin'):   #input pm, p, rect = 'sa','s',  mode = 'exact', 'bin'
    # set up
    if rect =='s' and mode =='lp':
        return lps1
    if rect =='sa' and p=='inf' and mode=='lp':
        return lpsainf 
    if rect == 'sa' and p==1 and mode =='lp':
        return lpsa1
    if rect=='nr':
        # print('Non robust',nr)
        return nr
    if p==1 and rect == 's': 
        # print('s1 exact',s1)
        return s1
    if p==2 and rect == 's' and (mode =='exact' or mode=='auto'):
        # print('s2 exact',s2)
        return s2
    if p=='inf' and rect == 's':
        # print('sinf exact',sinf)
        return sinf
    if rect == 's' :
        # print('s{} bin'.format(p),sp)
        return sp
    if rect == 'sa': 
        # print('sa{} bin'.format(p),sap)
        return sap
    else: 
        print('error in setup: NO {} rectangular L_{} robust in {} mode '.format(rect, p,mode)) 
        return None
     
def RVI(v,pm,p=2, rect='sa', mode ='exact', tol=0.0001):
    VI = setup(p=p,rect=rect,mode=mode) # value iteration function set
    return VI(v=v,pm=pm,p=p,tol=tol,mode=mode)

        
    